# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from .colorspace import bgr2rgb, rgb2bgr


def imnormalize(img, mean, std, to_rgb=True):
    img = img.astype(np.float32)
    mean = mean.reshape(1, 3).astype(np.float64)
    std = std.reshape(1, 3).astype(np.float64)
    if to_rgb:
        img = bgr2rgb(img)
    return cv2.divide(cv2.subtract(img, mean), std)


def imdenormalize(img, mean, std, to_bgr=True):
    img = (img * std) + mean
    if to_bgr:
        img = rgb2bgr(img)
    return img
