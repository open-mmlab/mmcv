# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np

from .colorspace import bgr2rgb, rgb2bgr


def imnormalize(img, mean, std, to_rgb=True):
    img = np.float32(img)
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        img = bgr2rgb(img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    img = cv2.add(img, mean, img)  # inplace
    if to_bgr:
        img = rgb2bgr(img)
    return img
