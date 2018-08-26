from __future__ import division

import cv2
import numpy as np

from .io import read_img


def bgr2gray(img, keepdim=False):
    """Convert a BGR image to grayscale image.

    Args:
        img (ndarray or str): The input image or image path.
        keepdim (bool): If False (by default), then return the grayscale image
            with 2 dims, otherwise 3 dims.

    Returns:
        ndarray: The converted grayscale image.
    """
    in_img = read_img(img)
    out_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    if keepdim:
        out_img = out_img[..., None]
    return out_img


def gray2bgr(img):
    """Convert a grayscale image to BGR image.

    Args:
        img (ndarray or str): The input image or image path.

    Returns:
        ndarray: The converted BGR image.
    """
    in_img = read_img(img)
    in_img = in_img[..., None] if in_img.ndim == 2 else in_img
    out_img = cv2.cvtColor(in_img, cv2.COLOR_GRAY2BGR)
    return out_img


def convert_color_factory(src, dst):

    code = getattr(cv2, 'COLOR_{}2{}'.format(src.upper(), dst.upper()))

    def convert_color(img):
        in_img = read_img(img)
        out_img = cv2.cvtColor(in_img, code)
        return out_img

    convert_color.__doc__ = """Convert a {0} image to {1} image

    Args:
        img (ndarray or str): The input image or image path.

    Returns:
        ndarray: The converted {1} image
    """.format(src.upper(), dst.upper())

    return convert_color


bgr2rgb = convert_color_factory('bgr', 'rgb')

rgb2bgr = convert_color_factory('rgb', 'bgr')

bgr2hsv = convert_color_factory('bgr', 'hsv')

hsv2bgr = convert_color_factory('hsv', 'bgr')


def scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple): w, h.
        scale (float): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def resize(img, size, return_scale=False, interpolation='bilinear'):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image or image path.
        size (tuple): Target (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    img = read_img(img)
    h, w = img.shape[:2]
    resized_img = cv2.resize(
        img, size, interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / float(w)
        h_scale = size[1] / float(h)
        return resized_img, w_scale, h_scale


def resize_like(img, dst_img, return_scale=False, interpolation='bilinear'):
    """Resize image to the same size of a given image.

    Args:
        img (ndarray): The input image or image path.
        dst_img (ndarray): The target image.
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    h, w = dst_img.shape[:2]
    return resize(img, (w, h), return_scale, interpolation)


def resize_by_ratio(img, ratio, interpolation='bilinear'):
    """Resize image by a ratio.

    Args:
        img (ndarray): The input image or image path.
        ratio (float): Scaling factor.
        interpolation (str): Same as :func:`resize`.

    Returns:
        ndarray: The resized image
    """
    assert isinstance(ratio, (float, int)) and ratio > 0
    img = read_img(img)
    h, w = img.shape[:2]
    new_size = scale_size((w, h), ratio)
    return resize(img, new_size, interpolation=interpolation)


def resize_keep_ar(img,
                   max_long_edge,
                   max_short_edge,
                   return_scale=False,
                   interpolation='bilinear'):
    """Resize image with aspect ratio unchanged.

    The long edge of resized image will be no greater than `max_long_edge`,
    and the short edge of resized image is no greater than `max_short_edge`.

    Args:
        img (ndarray): The input image or image path.
        max_long_edge (int): Max value of the long edge of resized image.
        max_short_edge (int): Max value of the short edge of resized image.
        return_scale (bool): Whether to return scale besides the resized image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (resized_img, scale_factor) or resized_img.
    """
    if max_long_edge < max_short_edge:
        raise ValueError(
            '"max_long_edge" should not be less than "max_short_edge"')
    img = read_img(img)
    h, w = img.shape[:2]
    ratio = min(
        float(max_long_edge) / max(h, w),
        float(max_short_edge) / min(h, w))
    new_size = scale_size((w, h), ratio)
    resized_img = resize(img, new_size, interpolation=interpolation)
    if return_scale:
        return resized_img, ratio
    else:
        return resized_img


def limit_size(img, max_edge, return_scale=False, interpolation='bilinear'):
    """Limit the size of an image.

    If the long edge of the image is greater than max_edge, resize the image.

    Args:
        img (ndarray): The input image or image path.
        max_edge (int): Maximum value of the long edge.
        return_scale (bool): Whether to return scale besides the resized image.
        interpolation (str): Same as :func:`resize`.

    Returns:
        tuple or ndarray: (resized_img, scale_factor) or resized_img.
    """
    img = read_img(img)
    h, w = img.shape[:2]
    if max(h, w) > max_edge:
        scale = float(max_edge) / max(h, w)
        new_size = scale_size((w, h), scale)
        resized_img = resize(img, new_size, interpolation=interpolation)
    else:
        scale = 1.0
        resized_img = img
    if return_scale:
        return resized_img, scale
    else:
        return resized_img


def bbox_clip(bboxes, img_shape):
    """Clip bboxes to fit the image shape.

    Args:
        bboxes (ndarray): Shape (..., 4*k)
        img_shape (tuple): (height, width) of the image.

    Returns:
        ndarray: Clipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    cliped_bboxes = np.empty_like(bboxes, dtype=bboxes.dtype)
    cliped_bboxes[..., 0::4] = np.maximum(
        np.minimum(bboxes[..., 0::4], img_shape[1] - 1), 0)
    cliped_bboxes[..., 1::4] = np.maximum(
        np.minimum(bboxes[..., 1::4], img_shape[0] - 1), 0)
    cliped_bboxes[..., 2::4] = np.maximum(
        np.minimum(bboxes[..., 2::4], img_shape[1] - 1), 0)
    cliped_bboxes[..., 3::4] = np.maximum(
        np.minimum(bboxes[..., 3::4], img_shape[0] - 1), 0)
    return cliped_bboxes


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


def crop_img(img, bboxes, scale_ratio=1.0, pad_fill=None):
    """Crop image patches.

    3 steps: scale the bboxes -> clip bboxes -> crop and pad.

    Args:
        img (ndarray): Image to be cropped.
        bboxes (ndarray): Shape (k, 4) or (4, ), location of cropped bboxes.
        scale_ratio (float, optional): Scale ratio of bboxes, the default value
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
    img = read_img(img)
    _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
    scaled_bboxes = bbox_scaling(_bboxes, scale_ratio).astype(np.int32)
    clipped_bbox = bbox_clip(scaled_bboxes, img.shape)
    patches = []
    for i in range(clipped_bbox.shape[0]):
        x1, y1, x2, y2 = tuple(clipped_bbox[i, :].tolist())
        if pad_fill is None:
            patch = img[y1:y2 + 1, x1:x2 + 1, ...]
        else:
            _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :].tolist())
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


def pad_img(img, shape, pad_val):
    """Pad an image to a certain shape.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or list): Values to be filled in padding areas.

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


def rotate_img(img,
               angle,
               center=None,
               scale=1.0,
               border_value=0,
               auto_bound=False):
    """Rotate an image.

    Args:
        img (ndarray or str): Image to be rotated.
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
    img = read_img(img)
    h, w = img.shape[:2]
    if center is None:
        center = ((w - 1) / 2, (h - 1) / 2)
    assert isinstance(center, tuple)
    matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    if auto_bound:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        matrix[0, 2] += (new_w - w) / 2
        matrix[1, 2] += (new_h - h) / 2
        w = int(np.round(new_w))
        h = int(np.round(new_h))
    rotated = cv2.warpAffine(img, matrix, (w, h), borderValue=border_value)
    return rotated
