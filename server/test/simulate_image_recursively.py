import base64
import json
import os
import requests
import time

path = "id_test/Under_21/images/"
fileList = os.listdir(path)

failed_files = []

for file in fileList:
    startTime = time.time()
    file = os.path.join(path + file)
    with open(file, "rb") as img:
        img = base64.b64encode(img.read())
    img_from_post = img.decode(
        "ascii"
    )  # Simulate the image data received from frontend.


    url = "http://localhost:8282/scan"
    headers = {"Scan_OCR": "AdviNowMed13371337", "Content-Type": "application/json"}

    payload = {
        "image": img_from_post,
        "aws_access_key_id": "AKIATXVAKOSYQPPJYCGH",
        "aws_secret_access_key": "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",
        "single_m": 0.03,
        "multi_m": 0.07,
    }
    with requests.Session() as session:
        response = session.post(url, headers=headers, json=payload)
        try:
            print(response.json())
        except:
            print(file + " - OCR failed!")
            failed_files.append(file[15:-4])
        totalTime = time.time() - startTime
        print("Runtime: {:.2f} seconds.".format(totalTime))

    with open("id_test/Under_21/json_ocr_mapping/" + file[24:-4] + ".json", "w") as f:
        if response.status_code == 200:
            json.dump(response.json(), f, indent=4, sort_keys=True)

print(failed_files)
