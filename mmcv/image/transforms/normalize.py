# Copyright (c) Open-MMLab. All rights reserved.
import cv2
import numpy as np


def imnormalize(img, mean, std, to_rgb=True):
    img = np.float32(img) if img.dtype != np.float32 else img.copy()
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    print(img[0, 0])
    return img
