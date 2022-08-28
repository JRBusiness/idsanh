import json
import os
import sys

import mmcv
import copy
import structlog
import numpy as np
import cv2

from server.shared.helpers.ppocr.utils import utility
from server.shared.helpers.ppocr.utils.utility import check_and_read_gif
from server.shared.helpers.tools.infer.predict_det import TextDetector
from server.shared.helpers.mmocr.utils.box_util import stitch_boxes_into_lines
from server.shared.helpers.mmocr.datasets.kie_dataset import KIEDataset
from server.shared.helpers.mmocr.datasets.pipelines.crop import crop_img
from server.shared.helpers.mmocr.apis.inference import model_inference
from server.shared.helpers.mmocr.utils.ocr import MMOCR
from server.logs import LogHandler
from server.shared.helpers.tools.infer.predict_rec import TextRecognizer
from settings import config


det_batch_size = 0
merge_xdist = 20
variable = LogHandler("ocr_engine")
structlog.configure(processors=[variable.event_dict, structlog.processors.JSONRenderer()])
structlog.PrintLoggerFactory(sys.stdout)
logger = structlog.get_logger()


def detection(image_file):
    text_detector = TextDetector(config)
    count = 0
    total_time = 0
    draw_img_save = "./inference_results"

    if config.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for i in range(2):
            res = text_detector(img)

    if not os.path.exists(draw_img_save):
        os.makedirs(draw_img_save)
    save_results = []
    img, flag = check_and_read_gif(image_file)
    if not flag:
        img = cv2.imread(image_file)
    dt_boxes, _ = text_detector(img)
    save_pred = os.path.basename(image_file) + "\t" + str(
        json.dumps([x.tolist() for x in dt_boxes])) + "\n"
    save_results.append(save_pred)
    logger.info(save_pred)
    src_im = utility.draw_text_det_res(dt_boxes, image_file)
    img_name_pure = os.path.split(image_file)[-1]
    img_path = os.path.join(draw_img_save,
                            "det_res_{}".format(img_name_pure))
    cv2.imwrite(img_path, src_im)
    logger.info("The visualized image saved in {}".format(img_path))

    with open(os.path.join(draw_img_save, "det_results.txt"), 'w') as f:
        f.writelines(save_results)
        f.close()


def recognition(image_file):
    text_recognizer = TextRecognizer(config)
    valid_image_file_list = []
    img_list = []
    img, flag = check_and_read_gif(image_file)
    if not flag:
        img = cv2.imread(image_file)
    valid_image_file_list.append(image_file)
    img_list.append(img)
    try:
        rec_res, _ = text_recognizer(img_list)
        return rec_res
    except Exception as E:
        logger.info(E)
        exit()


def key_info_extraction(inference_handler, arrays, image, filename, batch_mode, kie_model=None):
    det_result = detection(image)  # prepare the text detection model.
    bboxes_list = [res['boundary_result'] for res in det_result]
    kie_dataset = KIEDataset(
        dict_file=kie_model.cfg.data.test.dict_file)
    output = f"output_{filename}"
    for filename, arr, bboxes, out_file in zip(filename,
                                               arrays,
                                               bboxes_list,
                                               output):
        img_e2e_res = {}
        img_e2e_res['filename'] = filename
        img_e2e_res['result'] = []
        box_imgs = []
        for bbox in bboxes:
            box_res = {}
            box_res['box'] = [round(x) for x in bbox[:-1]]
            box_res['box_score'] = float(bbox[-1])
            box = bbox[:8]
            if len(bbox) > 9:
                min_x = min(bbox[0:-1:2])
                min_y = min(bbox[1:-1:2])
                max_x = max(bbox[0:-1:2])
                max_y = max(bbox[1:-1:2])
                box = [
                    min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y
                ]
            box_img = crop_img(arr, box)
            # Prepare the text recognition model
            if batch_mode:
                box_imgs.append(box_img)
            else:
                recog_result = recognition(image)
                text = recog_result['text']
                text_score = recog_result['score']
                if isinstance(text_score, list):
                    text_score = sum(text_score) / max(1, len(text))
                box_res['text'] = text
                box_res['text_score'] = text_score
                img_e2e_res['result'].append(box_res)
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

