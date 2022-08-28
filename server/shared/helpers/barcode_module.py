import cv2
import datetime
import io
import base64
import cv2.dnn
import structlog
import sys
import numpy as np

from PIL import Image as Pil
from pdf417decoder import PDF417Decoder
from ddtrace import patch_all, tracer

from server.logs import LogHandler
from settings import Config
from server.shared.helpers.schemas import Response

variable = LogHandler(Config.app_name)
structlog.configure(processors=[variable.event_dict, structlog.processors.JSONRenderer()])
patch_all()
structlog.PrintLoggerFactory(sys.stdout)
log = structlog.get_logger()


class OptionFields:
    first_middle = ("DCT", str)
    first_name = ("DAC", str)
    last_name = ("DCS", str)
    middle_name = ("DAD", str)
    address = ("DAG", str)
    address2 = ("DAH", str)
    city = ("DAI", str)
    state = ("DAJ", str)
    country = ("DCG", str)
    license_number = ("DAQ", str)
    expiry = ("DBA", datetime.date)
    dob = ("DBB", datetime.date)
    ZIP = ("DAK", str)
    DL_class = ("DCA", str)
    restrictions = ("DCB", str)
    endorsements = ("DCD", str)
    sex = ("DBC", str)
    height = ("DAU", str)
    weight = ("DCE", str)
    hair = ("DAZ", str)
    eyes = ("DAY", str)
    issued = ("DBD", datetime.date)
    document = ("DCF", str)
    revision = ("DDB", datetime.date)


class BarcodeHandler(OptionFields):
    def __init__(self):
        self.person = {}

    @tracer.wrap()
    def decode_dl(self, raw_data) -> Response:
        keys = list(
            filter(lambda x: not x[0].startswith("__"), vars(OptionFields).items())
        )
        for line in raw_data.splitlines():
            for value_name, (start_sequence, key_type) in keys:
                if line.startswith(start_sequence):
                    data = line[len(start_sequence), ]
                    if key_type == datetime.date:
                        data = datetime.datetime.strptime(data, "%m%d%Y").date()
                    if data != "":
                        self.person[value_name] = data
        return Response(
            success=True,
            response={
                # "face_photo": face,
                **{k: v.replace(" ", "") for k, v in self.person.items()},
            },
        )

    @tracer.wrap()
    def select_barcode_region(self, input_image):
        img_stream = io.BytesIO(input_image)
        image = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
        image_cp = image.copy()
        image = cv2.pyrMeanShiftFiltering(image, 25, 25, 50)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(grad_x, grad_y)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closing_kern_odd = 1
        kernel_closing = np.ones((closing_kern_odd, closing_kern_odd), np.uint8)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_closing)
        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_count = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_count)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if rect[2] != 90:
            image_cp = cv2.drawContours(image_cp, [box], 0, (0, 0, 255), 2)
        else:
            x, y, w, h = cv2.boundingRect(max_count)
            image_cp = cv2.rectangle(image_cp, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image_cp

    @tracer.wrap()
    def sharpen_image(self, image):
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, sharpen_kernel)

    @tracer.wrap()
    def get_raw_data(self, images):
        image = Pil.open(io.BytesIO(images))
        decoder = PDF417Decoder(image)
        if decoder.decode() <= 0:
            image = image.copy()
            # print("#"*30)
            # print("Checking with the rotation of image ...")
            for _ in range(3):
                # print(f"Rotating image anticlockwise -- {i+1}")
                im1 = image.rotate(90, Pil.NEAREST, expand=True)
                decoder = PDF417Decoder(im1)
                if decoder.decode() > 0:
                    return decoder.barcode_data_index_to_string(0)
                else:
                    image = im1
            barcode_region = self.select_barcode_region(images)
            image = Pil.fromarray(barcode_region)
            decoder = PDF417Decoder(image)
            if decoder.decode() <= 0:
                return None
        return decoder.barcode_data_index_to_string(0)

    @tracer.wrap()
    def decode_bar(self, img_data):
        bs64_img = base64.b64decode(img_data)
        raw_data = self.get_raw_data(bs64_img)

        if raw_data is not None:
            return self.decode_dl(raw_data)
