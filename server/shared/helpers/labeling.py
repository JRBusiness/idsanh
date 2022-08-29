import mmcv
import copy

from mmocr.datasets.kie_dataset import KIEDataset
from mmocr.apis.inference import model_inference
from mmocr.utils.ocr import MMOCR
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
det_batch_size = 0
merge_xdist = 20


def get_box(array):
    result = []
    for i in array:
        for a in i:
            result.append(a)
    return result


def key_info_extraction(inference_handler, arrays, filename, kie_model=None):
    try:
        kie_dataset = KIEDataset(
            dict_file=kie_model.cfg.data.test.dict_file)
        output = f"output_{filename}"
        img_e2e_res = {}
        img_e2e_res['filename'] = filename
        img_e2e_res['result'] = []
        recog_results = ocr.ocr(filename, cls=True)
        boxes = [line[0] for line in recog_results]
        txts = [line[1][0] for line in recog_results]
        scores = [line[1][1] for line in recog_results]
        # im_show = draw_ocr(filename, boxes, txts, scores, font_path='server/shared/helpers/kie/utils/simfang.ttf')
        # im_show = Image.fromarray(im_show)
        # im_show.save("output/image.png")
        # im_show.show()
        for recog_result in recog_results:
            box_res = {}
            array_box = recog_result[0]
            box_res['box'] = get_box(array_box)
            text_score = recog_result[1][1]
            text = recog_result[1][0]
            if isinstance(text_score, list):
                text_score = sum(text_score) / max(1, len(text))
            box_res['box_score'] = box_res['text_score'] = text_score
            box_res['text'] = text
            img_e2e_res['result'].append(box_res)

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
            arrays,
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
    except Exception as e:
        raise(e)


def ocr_module(filename):  # Executing the ocr module ....
    arrays = [mmcv.imread(filename)]
    inference_handler = MMOCR(
        kie='SDMGR',  # Our piece of cheese that handle the key extraction.
        kie_config="server/shared/helpers/kie/config.py",
        kie_ckpt="server/shared/helpers/kie/checkpoint.pth",
    )
    return key_info_extraction(
        inference_handler,
        arrays,
        filename,
        kie_model=inference_handler.kie_model,
    )
