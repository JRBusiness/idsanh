import base64
import io
import json
import pathlib
import re
import sys
from glob import glob
from time import time

import cv2
import structlog
import os
import numpy as np

from settings import Config
from server.shared.helpers.state_adjust import AdjustHandler
from server.logs import LogHandler
from server.shared.helpers.labeling import ocr_module


class InsuranceCard:
    def __init__(self, det, recog):
        self.det = det
        self.recog = recog

    def image_processing(self, imagedata):
        imgdata = base64.b64decode(imagedata)
        stream_img = io.BytesIO(imgdata)
        images = cv2.imdecode(np.frombuffer(stream_img.read(), np.uint8), 1)
        return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

    def ocr_image(self, image):
        result_dict = {}
        results = ocr_module(image, self.det, self.recog)["result"]
        labels = set()
        for result in results:
            if result["label"] in labels:
                if result["label_score"] >= 0.98:
                    result_dict[result["label"]].append(result["text"])
            elif result["label"] not in labels:
                labels.add(result["label"])
                result_dict[result["label"]] = [[result["text"]], ]
        return result_dict


def scan_insurance():
    det = [
        "PANet_CTW",
    ]
    for i in det:
        det = i
        recog = "CRNN"
        results = []
        handler = InsuranceCard(det, recog)
        for file in glob("/home/jerrydev/Desktop/mmocr/tests/wildreceipt/validate/*.jpg"):
            with open(file, "rb") as imgage:
                img = base64.b64encode(imgage.read())
                img_from_post = img.decode("ascii")
                image = handler.image_processing(img_from_post)
                result = handler.ocr_image(image)
                results.append(result)
                print(result)
        with open(f"root/Desktop/test_result{det}_{recog}.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    scan_insurance()