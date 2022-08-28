import base64
import sys
import requests
import csv
from os import path
import glob
import os
import json
import time

failed_files = []
path = "server/test/id_test/Above_21/images"
fileList = os.listdir(path)

url = "http://id-scanner.advinow-dev.int/scan_id"
for file in sorted(glob.glob(os.path.join(path, "*.jpg"))):
    startTime = time.time()
    # while arguments >= position:
    with open(file, "rb") as imgage:
        img = base64.b64encode(imgage.read())
    img_from_post = img.decode("ascii")  # Simulate the image data received from frontend.
    # headers = {"Scan_OCR": "AdviNowMed13371337", "Content-Type": "octet}

    with requests.Session() as session:
        response = session.post(
            url,
            headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer a3eee57d-9c46-4e30-aae6-c2c13710ccc6'
            },
            json={
                "img_data": img_from_post
            })
        try:
            print(response.json())
        except Exception as e:
            print(f'{file} - OCR failed!')
