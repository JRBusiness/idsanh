import base64
import io
import pathlib
import re
import sys
from glob import glob

import boto3
import structlog
import cv2
import os
import torch
import numpy as np
import torchvision
import json

from PIL import Image
from lxml.etree import Element, ElementTree
from matplotlib import pyplot as plt
from py_linq import Enumerable
import xml.etree.ElementTree as ET
from collections import defaultdict
from fastai.learner import load_learner
from ddtrace import patch_all, tracer

from server.shared.helpers.processing import unwarp
from settings import Config
from server.shared.helpers.state_adjust import AdjustHandler
from server.logs import LogHandler
from server.shared.helpers.schemas import Response, TemplateResponse

variable = LogHandler(Config.app_name)
structlog.configure(processors=[variable.event_dict, structlog.processors.JSONRenderer()])
patch_all()
structlog.PrintLoggerFactory(sys.stdout)
log = structlog.get_logger()

state_list = [
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinoisIndiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montanaNebraska",
    "nevada",
    "newhampshire",
    "newjersey",
    "newmexico",
    "newyork",
    "northcarolina",
    "northdakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhodeisland",
    "southcarolina",
    "southdakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west Virginia",
    "wisconsin",
    "wyoming"
]


def adjustment_by_state(result_dict: dict, goods: dict):
    adjust = AdjustHandler(goods)
    if result_dict['state_name'].lower() == 'alabama':
        result_dict = adjust.alabama(result_dict)
    if result_dict['state_name'].lower() == "california":
        result_dict = adjust.california(result_dict)
    if result_dict['state_name'].lower() == 'kansas':
        result_dict = adjust.kansas(result_dict)
    return result_dict


def find_by_regex(result_dict: dict):
    addy_string = re.compile(r'^(\d+\W?|\W?)\s\d')
    name_string = re.compile(r'^(\W|\d+\w?|\s?)\s?\w')
    date_string = re.compile(r'.*?(\d{2}/?-?\d{2}/?-?\d{4})')
    for key, value in result_dict.items():
        if key in ["address", "address2", "address3", "address4"]:
            if addy_string.match(value):
                result_dict[key] = value.strip(addy_string.search(value)[1]).strip(r'^ ')
        elif name_string.match(value):
            result_dict[key] = re.search(
                date_string,
                value
            )[1] if date_string.match(value) else value.strip(name_string.search(value)[1]).strip(r'^ ')
    return result_dict


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = width_b = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = height_b = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


class IDHandler:
    def __init__(self, proto_text, caffe_model, id_file):
        if os.name == 'nt':
            pathlib.PosixPath = pathlib.WindowsPath
        self.learner = load_learner(id_file)
        self.aws_key = Config.aws_key
        self.aws_id = Config.aws_id
        # self.face = cv2.dnn.readNetFromCaffe(proto_text, caffe_model)
        with open('server/api/models/class_mapping.json') as data:
            mappings = json.load(data)
        self.class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load('server/api/models/model.pt').to(self.device)

    @tracer.wrap()
    def get_prediction(self, img):
        image = np.array(img)
        h, w = image.shape[:2]
        x = torch.from_numpy(image).to(self.device)
        print(x.size())
        with torch.no_grad():
            x = x.permute(2, 0, 1).float()
            y = self.model(x)
            to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.3)
            y['pred_boxes'] = y['pred_boxes'][to_keep]
            y['pred_classes'] = y['pred_classes'][to_keep]
            y['pred_masks'] = y['pred_masks'][to_keep]

            all_masks = np.zeros((h, w), dtype=np.int8)
            instance_idx = 1
            for mask, bbox, label in zip(reversed(y['pred_masks']),
                                         y['pred_boxes'],
                                         y['pred_classes']):

                bbox = list(map(int, bbox))
                x1, y1, x2, y2 = bbox
                class_idx = label.item()
                class_name = self.class_mapping[class_idx]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(
                    image,
                    class_name,
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 0, 0)
                )
                all_masks[mask == 1] = instance_idx

                image_crop = image[y1:y2, x1:x2]
                instance_idx += 1

        plt.imshow(image_crop)
        # plt.imshow(all_masks, alpha=0.5)
        plt.show()

    @tracer.wrap()
    def analyze_id(self, imagedata):
        success, img = cv2.imencode(".jpg", imagedata)
        imagedata = img.tobytes()
        client = boto3.client(
            "textract",
            region_name="us-west-2",
            aws_access_key_id=Config.aws_id,
            aws_secret_access_key=Config.aws_key,
        )
        response = client.analyze_id(DocumentPages={"bytes": imagedata})
        result = {}
        for id_field in [doc_fields['IdentityDocumentFields'] for doc_fields in response['IdentityDocuments']]:
            result_type = ""
            result_value = ""
            for key, val in id_field.items():
                if "Type" in str(key):
                    result_type = "Type: " + str(val['Text'])
                if "ValueDetection" in str(key):
                    result_value = "Value Detection: " + str(val['Text'])
                result[result_type] = result_value
            return result

    @tracer.wrap()
    def extract_face(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face.setInput(blob)
        detections = self.face.forward()
        # Identify each face
        for i in range(detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            confidence = detections[0, 0, i, 2]
            # If confidence > 0.5, save it as a separate file
            if confidence > 0.5:
                return image[start_y:end_y, start_x:end_x]

    @tracer.wrap()
    def textract(self, imagedata):
        success, img = cv2.imencode(".jpg", imagedata)
        imagedata = img.tobytes()
        client = boto3.client(
            "textract",
            region_name="us-west-2",
            aws_access_key_id=self.aws_id,
            aws_secret_access_key=self.aws_key,
        )
        response: dict = client.detect_document_text(Document={"Bytes": imagedata})
        return response["Blocks"]

    @tracer.wrap()
    def get_template(self, name: str) -> TemplateResponse:
        tree: ElementTree = ET.parse(f"server/api/models/ID_template/{name}.xml")
        root: Element = tree.getroot()
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        pixels = {}
        # for root_object in list(ET.ElementTree(root).iterfind('object')):
        #     object_data = Enumerable(root_object.getchildren())
        #     position_data = list(ET.ElementTree(object_data.take(4).last()).iter())
        #     pixels: List[Pixel] = [Pixel(**{str(x): float(x.text)}) for x in Enumerable(position_data).skip(1).take(4)]
        # return TemplateResponse(pixels=pixels, width=width, height=height)
        ET.tostring(root, encoding="utf8").decode("utf8")
        for movie in root.iter("object"):
            name = movie[0].text
            x_min = int(movie[4][0].text)
            y_min = int(movie[4][1].text)
            x_max = int(movie[4][2].text)
            y_max = int(movie[4][3].text)
            pixels[name] = [x_min, y_min, x_max, y_max]
        return TemplateResponse(pixels=pixels, width=width, height=height)

    @tracer.wrap()
    def bb_intersection_over_union(self, box_a, box_b):
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
        return inter_area / float(box_a_area + box_b_area - inter_area)

    @tracer.wrap()
    def blocks(self, blocks, rois, in_block, per, width, height):
        dict_aws = {}
        for block in blocks:
            if block['BlockType'] == in_block:
                box = block['Geometry']['BoundingBox']
                left = width * box['Left']
                top = height * box['Top']
                right = left + (width * box['Width'])
                bottom = top + (height * box['Height'])
                text = block['Text']
                dict_aws[text] = [
                    int(left),
                    int(top),
                    int(right),
                    int(bottom)
                ]
        iou = []
        final_dict = {}
        for aws_keys, aws_values in dict_aws.items():
            for k, v in rois.items():
                iou.append(self.bb_intersection_over_union(v, aws_values))
            if max(iou) > per:
                final_dict[aws_keys] = [max(iou), list(rois)[np.argmax(iou)]]
            iou = []
        return dict(sorted(final_dict.items(), key=lambda item: item[1]))

    @tracer.wrap()
    def parser(self, image, name, template):
        blocks = self.textract(image)
        final_dict = self.blocks(blocks, template.pixels, "WORD", Config.single_m, template.width, template.height)
        result_dict = {}
        pattern = r'[0-9]'
        for k, v in final_dict.items():
            if k == "CLASS":
                k = "class_type"
            result_dict[v[1].lower()] = k
            if v == 'Last_Name':
                result_dict[v[1].lower()] = re.sub(pattern, '', k)
        name = name.split('_')[0]
        if any(name == state for state in state_list):
            goods = defaultdict(list)
            final_dict2 = self.blocks(blocks, template.pixels, "LINE", Config.multi_m, template.width, template.height)
            for k, v in final_dict2.items():
                goods[v[1]].append(k)
                if name not in ['colorado_u21', 'michigan_u21']:
                    result_dict['first_name'] = re.sub(pattern, '', ''.join(goods['first_name']))
                result_dict['id_type'] = ''.join(goods['id_type'])
                result_dict['address'] = ''.join(goods['address'])
                result_dict['city_state_zip'] = ''.join(goods['city_state_zip'])
                result_dict['state_name'] = ''.join(goods['state_name'])
                result_dict = adjustment_by_state(result_dict, goods)
            return find_by_regex(result_dict)

    @tracer.wrap()
    def process_img(self, imagedata):
        imgdata = base64.b64decode(imagedata)
        img_stream = io.BytesIO(imgdata)
        images = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    id_model = "server/api/models/ID.pkl"
    is_model = "server/api/models/insurance.pkl"
    proto_text = "server/api/models/config.prototxt"
    caffe_model = "server/api/models/trained.caffemodel"
    handler = IDHandler(proto_text, caffe_model, id_model)
    for file in glob("server/test/id_test/insurance/sample/*.jpg"):
        image = cv2.imread(file, 0)
        w, h = image.shape[0], image.shape[1]
        src = np.float32([(20, 1),
                          (540, 130),
                          (20, 520),
                          (570, 450)])

        dst = np.float32([(600, 0),
                          (0, 0),
                          (600, 531),
                          (0, 531)])

        image = unwarp(image, src, dst, True)
        handler.main(image)
