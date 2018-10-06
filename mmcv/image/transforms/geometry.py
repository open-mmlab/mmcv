from __future__ import division

import cv2
import numpy as np


def imflip(img, direction='horizontal'):
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or "vertical".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    else:
        return np.flip(img, axis=0)


def imrotate(img,
             angle,
             center=None,
             scale=1.0,
             border_value=0,
             auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        center (tuple): Center of the rotation in the source image, by default
            it is the center of the image.
        scale (float): Isotropic scale factor.
        border_value (int): Border value.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image.

    Returns:
        ndarray: The rotated image.
    """
    if center is not None and auto_bound:
        raise ValueError('`auto_bound` conflicts with `center`')
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) * 0.5, (h - 1) * 0.5)
    assert isinstance(center, tuple)

    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) * 0.5
        matrix[1, 2] += (new_h - h) * 0.5
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated


def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    clipped_bboxes = np.empty_like(bboxes, dtype=bboxes.dtype)
    clipped_bboxes[..., 0::2] = np.maximum(
        np.minimum(bboxes[..., 0::2], img_shape[1] - 1), 0)
    clipped_bboxes[..., 1::2] = np.maximum(
        np.minimum(bboxes[..., 1::2], img_shape[0] - 1), 0)
    return clipped_bboxes


def bbox_scaling(bboxes, scale, clip_shape=None):
    """Scaling bboxes w.r.t the box center.

    Args:
        bboxes (ndarray): Shape(..., 4).
        scale (float): Scaling factor.
        clip_shape (tuple, optional): If specified, bboxes that exceed the
            boundary will be clipped according to the given shape (h, w).

    Returns:
        ndarray: Scaled bboxes.
    """
    if float(scale) == 1.0:
        scaled_bboxes = bboxes.copy()
    else:
        w = bboxes[..., 2] - bboxes[..., 0] + 1
        h = bboxes[..., 3] - bboxes[..., 1] + 1
        dw = (w * (scale - 1)) * 0.5
        dh = (h * (scale - 1)) * 0.5
        scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
    if clip_shape is not None:
        return bbox_clip(scaled_bboxes, clip_shape)
    else:
        return scaled_bboxes


def imcrop(img, bboxes, scale=1.0, pad_fill=None):
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale (float, optional): Scale ratio of bboxes, the default value
            1.0 means no padding.
        pad_fill (number or list): Value to be filled for padding, None for
            no padding.

    Returns:
        list or ndarray: The cropped image patches.
    """
    chn = 1 if img.ndim == 2 else img.shape[2]
    if pad_fill is not None:
        if isinstance(pad_fill, (int, float)):
            pad_fill = [pad_fill for _ in range(chn)]
        assert len(pad_fill) == chn

    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)

    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
            if chn == 2:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
            else:
                patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
            patch = np.array(
                pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
            x_start = 0 if _x1 >= 0 else -_x1
            y_start = 0 if _y1 >= 0 else -_y1
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            patch[y_start:y_start + h, x_start:x_start +
                  w, ...] = img[y1:y1 + h, x1:x1 + w, ...]
        patches.append(patch)

    if bboxes.ndim == 1:
        return patches[0]
    else:
        return patches


def impad(img, shape, pad_val=0):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.

    Returns:
        ndarray: The padded image.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad


def impad_to_multiple(img, divisor, pad_val=0):
    """Pad an image to ensure each edge to be multiple to some number.

    Args:
        img (ndarray): Image to be padded.
        divisor (int): Padded image edges will be multiple to divisor.
        pad_val (number or sequence): Same as :func:`impad`.

    Returns:
        ndarray: The padded image.
    """
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, (pad_h, pad_w), pad_val)
