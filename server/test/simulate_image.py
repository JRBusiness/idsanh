import base64
import sys
import requests
import time

startTime = time.time()

# Count the arguments
arguments = len(sys.argv) - 1

# Output argument-wise
position = 1
while arguments >= position:
    with open(sys.argv[position], "rb") as img:
        img = base64.b64encode(img.read())
    img_from_post = img.decode(
        "ascii"
    )  # Simulate the image data received from frontend.
    position += 1

url = "https://scanner.wyinvestigative.com/scan"
headers = {"Scan_OCR": "AdviNowMed13371337", "Content-Type": "application/json"}

payload = {
    "image": img_from_post,  
    "aws_access_key_id": "AKIATXVAKOSYQPPJYCGH",
    "aws_secret_access_key": "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",
    "single_m": 0.03,
    "multi_m": 0.07
}
with requests.Session() as session:
    response = session.post(url, headers=headers, json=payload)
    print(response.json())
    totalTime = time.time() - startTime
    print('Runtime: {:.2f} seconds.'.format(totalTime))