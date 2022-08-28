import base64
import io
import pathlib
import re
import sys
import boto3
import cv2
import structlog
import os
import numpy as np

from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from fastai.learner import load_learner
from ddtrace import patch_all, tracer
from lxml.etree import Element
from py_linq import Enumerable

from settings import Config
from server.logs import LogHandler
from server.shared.helpers.schemas import Response, TemplateResponse


variable = LogHandler(Config.app_name)
structlog.configure(processors=[variable.event_dict, structlog.processors.JSONRenderer()])
patch_all()
structlog.PrintLoggerFactory(sys.stdout)
log = structlog.get_logger()


class InsuranceHandler:
    def __init__(self, ai_model):
        if os.name == 'nt':
            pathlib.PosixPath = pathlib.WindowsPath
        self.learner = load_learner(ai_model)
        self.aws_key = Config.aws_key
        self.aws_id = Config.aws_id

    @tracer.wrap()
    def bucket_name(self):
        s3 = boto3.resource('s3')
        for bucket in s3.buckets.all():
            return bucket.name

    @tracer.wrap()
    def textract(self, imagedata):
        success, img = cv2.imencode(".jpg", imagedata)
        imagedata = img.tobytes()
        client = boto3.client(
            "textract",
            region_name="us-west-2",
            aws_access_key_id=Config.aws_id,
            aws_secret_access_key=Config.aws_key,
        )
        response = client.detect_document_text(Document={"Bytes": imagedata})
        return response["Blocks"]

    @tracer.wrap()
    def get_template(self, name: str) -> TemplateResponse:
        tree: ElementTree = ET.parse(f"server/api/models/insurance_template/{name}.xml")
        root: Element = tree.getroot()
        width = int(root.find("./size/width").text)
        height = int(root.find("./size/height").text)
        pixels = {}
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
            result_dict[v[1]] = k
            if v == 'Last_Name':
                result_dict[v[1]] = re.sub(pattern, '', k)
        return result_dict

    @tracer.wrap()
    def process_img(self, imagedata):
        imgdata = base64.b64decode(imagedata)
        stream_img = io.BytesIO(imgdata)
        images = cv2.imdecode(np.frombuffer(stream_img.read(), np.uint8), 1)
        return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    tracer.wrap()

    def main(self, imgdata: str) -> Response:
        img_data = imgdata
        image = self.process_img(img_data)
        name = Enumerable(self.learner.predict(image)).first()
        template: TemplateResponse = self.get_template(name)
        image = cv2.resize(image, (template.width, template.height))
        return Response(
            success=True,
            response={
                # "face_photo": face,
                **self.parser(image, name, template),
            },
        )
