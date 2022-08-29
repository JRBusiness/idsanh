import mmcv
import copy

from mmocr.utils.box_util import stitch_boxes_into_lines
from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.apis.inference import model_inference
from mmocr.utils.ocr import MMOCR
from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='en')

det_batch_size = 0
merge_xdist = 20


def key_info_extraction(inference_handler, det_model, recog_model, arrays, filename, batch_mode, kie_model=None):
    # det_result = inference_handler.single_inference(
    #     det_model,
    #     arrays,
    #     False,
    #     det_batch_size
    # )  # prepare the text detection model.
    # bboxes_list = [res['boundary_result'] for res in det_result]
    # kie_dataset = KIEDataset(
    #     dict_file=kie_model.cfg.data.test.dict_file)
    # output = f"output_{filename}"
    # for filename, arr, bboxes, out_file in zip(filename,
    #                                            arrays,
    #                                            bboxes_list,
    #                                            output):
    #     img_e2e_res = {}
    #     img_e2e_res['filename'] = filename
    #     img_e2e_res['result'] = []
    #     box_imgs = []
    #     for bbox in bboxes:
    #         box_res = {}
    #         box_res['box'] = [round(x) for x in bbox[:-1]]
    #         box_res['box_score'] = float(bbox[-1])
    #         box = bbox[:8]
    #         if len(bbox) > 9:
    #             min_x = min(bbox[0:-1:2])
    #             min_y = min(bbox[1:-1:2])
    #             max_x = max(bbox[0:-1:2])
    #             max_y = max(bbox[1:-1:2])
    #             box = [
    #                 min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
    #             ]
    #         box_img = crop_img(arr, box)
    #         # Prepare the text recognition model
    #         if batch_mode:
    #             box_imgs.append(box_img)
    #         else:
    #             recog_result = model_inference(recog_model, box_img)
    #             text = recog_result['text']
    #             text_score = recog_result['score']
    #             if isinstance(text_score, list):
    #                 text_score = sum(text_score) / max(1, len(text))
    #             box_res['text'] = text
    #             box_res['text_score'] = text_score
    #             img_e2e_res['result'].append(box_res)
    #         #  THe Text detection result.
    #     if batch_mode:
    #         recog_results = inference_handler.single_inference(
    #             recog_model, box_imgs, True, 1)
    img_e2e_res = {}
    result = ocr.ocr(filename, cls=True)
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    for i, recog_result in enumerate(result):
        text = recog_result[1][0]
        text_score = recog_result[1][1]
        if isinstance(text_score, (list, tuple)):
            text_score = sum(text_score) / max(1, len(text))
        img_e2e_res['result'].append(recog_result[0])
        img_e2e_res['result'][i]['text'] = text
        img_e2e_res['result'][i]['text_score'] = text_score
        
    img_e2e_res['result'] = stitch_boxes_into_lines(
        img_e2e_res['result'], merge_xdist, 0.5)    

    annotations = copy.deepcopy(img_e2e_res['result'])
    for i, ann in enumerate(annotations):
        min_x = min(ann['box'][::2])
        min_y = min(ann['box'][1::2])
        max_x = max(ann['box'][::2])
        max_y = max(ann['box'][1::2])
        annotations[i]['box'] = [
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
        ]
    ann_info = kie_dataset._parse_anno_info(annotations)
    ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                            ann_info['bboxes'])
    ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                            ann_info['bboxes'])
    #  Set up the KIE model for key info extraction.
    kie_result, data = model_inference(
        kie_model,
        arr,
        ann=ann_info,
        return_data=True,
        batch_mode=False
    )
    gt_bboxes = data['gt_bboxes'].data.numpy().tolist()
    #  Get the annotations data from the dataset used to train the KIE model.
    labels = inference_handler.generate_kie_labels(kie_result, gt_bboxes,
                                            kie_model.class_list)
    for i in range(len(gt_bboxes)):
        img_e2e_res['result'][i]['label'] = labels[i][0]
        img_e2e_res['result'][i]['label_score'] = labels[i][1]
    return img_e2e_res


def ocr_module(filename, det, recog): # Executing the ocr module ....
    arrays = [mmcv.imread(filename)]
    inference_handler = MMOCR(
        det=det,  # There are many supported detection and recognition models,
        recog=recog,    # and these 2 give the highest accuracy.
        kie='SDMGR',  # Our piece of cheese that handle the key extraction.
        kie_config="server/shared/helpers/mmocr/config.py",
        kie_ckpt="server/shared/helpers/mmocr/cp.pth",
    )
    return key_info_extraction(
        inference_handler,
        inference_handler.detect_model,
        inference_handler.recog_model,
        arrays,
        filename,
        batch_mode=True,
        kie_model=inference_handler.kie_model,
    )

