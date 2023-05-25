
import cv2
import numpy as np


def _image_resize_short_edge(image, size, inter=cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if h < w:
        r = size / h
        dim = (int(w * r), size)
    elif h == w:
        dim = (size, size)
    else:
        r = size / w
        dim = (size, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def _image_center_crop(image, crop_w, crop_h):

    (h, w) = image.shape[:2]
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    return image[top:bottom, left:right]


def preprocess_frame(img, size, mean, std, to_rgb, pixValNorm=False):

    assert img.dtype != np.uint8

    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(std, list):
        std = np.array(std)

    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace

    img = _image_resize_short_edge(img, size)
    (h, w) = img.shape[:2]
    ratio = 4 / 3
    if h < w:
        crop_h = size
        crop_w = min(w, int(size * ratio))
    else:
        crop_w = size
        crop_h = min(h, int(size * ratio))
    img = _image_center_crop(img, crop_w, crop_h)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    if pixValNorm:
        img = img / 255.0

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    return img
