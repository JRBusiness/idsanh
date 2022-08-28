import sys
from glob import glob

import numpy as np
import cv2
from os import listdir
from os import makedirs
from os import path


def testContourValidity(contour, full_width, full_height):
    # Max countour width/height/area is 95% of whole image
    max_threshold = 0.95
    # Min contour width/height/area is 30% of whole image
    min_threshold = 0.3
    min_area = full_width * full_height * min_threshold
    max_area = full_width * full_height * max_threshold
    max_width = max_threshold * full_width
    max_height = max_threshold * full_height
    min_width = min_threshold * full_width
    min_height = min_threshold * full_height

    # Area
    size = cv2.contourArea(contour)
    if size < min_area:
        return False
    if size > max_area:
        return False

    # Width / Height
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (tl, tr, br, bl) = sort_points(box)
    box_width = int(((br[0] - bl[0]) + (tr[0] - tl[0])) / 2)
    box_height = int(((br[1] - tr[1]) + (bl[1] - tl[1])) / 2)
    if box_width < min_width:
        return False
    if box_height < min_height:
        return False
    if box_width > max_width:
        return False
    if box_height > max_height:
        return False

    return True


# Find largest square that is not whole image
def find_square(im, f):
    # Width and height for validity check
    h = np.size(im, 0)
    w = np.size(im, 1)
    # Grayscale and blur before trying to find contours
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)
    debug_image(blur, 'preprocess_gray', f)
    # Threshold and contours
    flag, thresh = cv2.threshold(blur, 115, 255, cv2.THRESH_BINARY)
    debug_image(thresh, 'preprocess_thresh', f)
    rectamgle = []
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Debug
    im_debug = im.copy()
    # Find largest contour which does not take full image size.
    max = None
    for x in contours:
        if testContourValidity(x, w, h):
            im_debug = cv2.drawContours(im_debug, [x], -1, (0, 255, 0), 3)
            if max is None or cv2.contourArea(max) < cv2.contourArea(x):
                max = x
            debug_image(im_debug, 'possible_contours', f)
            # Min area rectangle around that contour. This nicely finds corners as MTG cards are rounded
            rect = cv2.minAreaRect(max)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return box


# Some helper functions for distance calculations
def dpot(a, b):
    return (a - b) ** 2


def adist(a, b):
    return np.sqrt(dpot(a[0], b[0]) + dpot(a[1], b[1]))


def max_distance(a1, a2, b1, b2):
    dist1 = adist(a1, a2)
    dist2 = adist(b1, b2)
    if int(dist2) < int(dist1):
        return int(dist1)
    else:
        return int(dist2)


# Sort points clockwise starting from top left
def sort_points(pts):
    try:
        ret = np.zeros((4, 2), dtype="float32")
        sumF = pts.sum(axis=1)
        diffF = np.diff(pts, axis=1)
        ret[0] = pts[np.argmin(sumF)]
        ret[1] = pts[np.argmin(diffF)]
        ret[2] = pts[np.argmax(sumF)]
        ret[3] = pts[np.argmax(diffF)]
        return ret
    except Exception as e:
        print(e)


# Fix perspective
def fix_perspective(image, pts):
    (tl, tr, br, bl) = sort_points(pts)
    maxW = max_distance(br, bl, tr, tl)
    maxH = max_distance(tr, br, tl, bl)
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dst)
    fixed = cv2.warpPerspective(image, transform, (maxW, maxH))
    return fixed


def debug_image(img, extra_path, filename):
    fpath = "debug/" + extra_path + "/"
    if not path.isdir(fpath):
        makedirs(fpath)
    cv2.imwrite(fpath + filename, img)


if __name__ == "__main__":
    for file in glob("server/shared/helpers/test/*.jpg"):
        try:
            img = cv2.imread(file)
            width = int(img.shape[1] * 20 / 100)
            height = int(img.shape[0] * 20 / 100)
            img = cv2.resize(img, (width, height))
            square = find_square(img, file)
            img = fix_perspective(img, square)
            cv2.imshow("image", img)
            cv2.waitKey(0)
            filename2 = f"perspective_fix/{file}"
            cv2.imwrite(filename2, img)
        except Exception as e:
            print(e)