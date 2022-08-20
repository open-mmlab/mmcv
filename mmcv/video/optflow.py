# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple, Union

import cv2
import numpy as np
from mmengine.utils import is_str

from mmcv.arraymisc import dequantize, quantize
from mmcv.image import imread, imwrite


def flowread(flow_or_path: Union[np.ndarray, str],
             quantize: bool = False,
             concat_axis: int = 0,
             *args,
             **kwargs) -> np.ndarray:
    """Read an optical flow map.

    Args:
        flow_or_path (ndarray or str): A flow map or filepath.
        quantize (bool): whether to read quantized pair, if set to True,
            remaining args will be passed to :func:`dequantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.

    Returns:
        ndarray: Optical flow represented as a (h, w, 2) numpy array
    """
    if isinstance(flow_or_path, np.ndarray):
        if (flow_or_path.ndim != 3) or (flow_or_path.shape[-1] != 2):
            raise ValueError(f'Invalid flow with shape {flow_or_path.shape}')
        return flow_or_path
    elif not is_str(flow_or_path):
        raise TypeError(f'"flow_or_path" must be a filename or numpy array, '
                        f'not {type(flow_or_path)}')

    if not quantize:
        with open(flow_or_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise OSError(f'Invalid flow file: {flow_or_path}')
            else:
                if header != 'PIEH':
                    raise OSError(f'Invalid flow file: {flow_or_path}, '
                                  'header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
    else:
        assert concat_axis in [0, 1]
        cat_flow = imread(flow_or_path, flag='unchanged')
        if cat_flow.ndim != 2:
            raise OSError(
                f'{flow_or_path} is not a valid quantized flow file, '
                f'its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)

    return flow.astype(np.float32)


def flowwrite(flow: np.ndarray,
              filename: str,
              quantize: bool = False,
              concat_axis: int = 0,
              *args,
              **kwargs) -> None:
    """Write optical flow to file.

    If the flow is not quantized, it will be saved as a .flo file losslessly,
    otherwise a jpeg image which is lossy but of much smaller size. (dx and dy
    will be concatenated horizontally into a single image if quantize is True.)

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        filename (str): Output filepath.
        quantize (bool): Whether to quantize the flow and save it to 2 jpeg
            images. If set to True, remaining args will be passed to
            :func:`quantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.
    """
    if not quantize:
        with open(filename, 'wb') as f:
            f.write(b'PIEH')
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert concat_axis in [0, 1]
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        imwrite(dxdy, filename)


def quantize_flow(flow: np.ndarray,
                  max_val: float = 0.02,
                  norm: bool = True) -> tuple:
    """Quantize flow to [0, 255].

    After this step, the size of flow will be much smaller, and can be
    dumped as jpeg images.

    Args:
        flow (ndarray): (h, w, 2) array of optical flow.
        max_val (float): Maximum value of flow, values beyond
                        [-max_val, max_val] will be truncated.
        norm (bool): Whether to divide flow values by image width/height.

    Returns:
        tuple[ndarray]: Quantized dx and dy.
    """
    h, w, _ = flow.shape
    dx = flow[..., 0]
    dy = flow[..., 1]
    if norm:
        dx = dx / w  # avoid inplace operations
        dy = dy / h
    # use 255 levels instead of 256 to make sure 0 is 0 after dequantization.
    flow_comps = [
        quantize(d, -max_val, max_val, 255, np.uint8) for d in [dx, dy]
    ]
    return tuple(flow_comps)


def dequantize_flow(dx: np.ndarray,
                    dy: np.ndarray,
                    max_val: float = 0.02,
                    denorm: bool = True) -> np.ndarray:
    """Recover from quantized flow.

    Args:
        dx (ndarray): Quantized dx.
        dy (ndarray): Quantized dy.
        max_val (float): Maximum value used when quantizing.
        denorm (bool): Whether to multiply flow values with width/height.

    Returns:
        ndarray: Dequantized flow.
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)

    dx, dy = (dequantize(d, -max_val, max_val, 255) for d in [dx, dy])

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def flow_warp(img: np.ndarray,
              flow: np.ndarray,
              filling_value: int = 0,
              interpolate_mode: str = 'nearest') -> np.ndarray:
    """Use flow to warp img.

    Args:
        img (ndarray): Image to be warped.
        flow (ndarray): Optical Flow.
        filling_value (int): The missing pixels will be set with filling_value.
        interpolate_mode (str): bilinear -> Bilinear Interpolation;
                                nearest -> Nearest Neighbor.

    Returns:
        ndarray: Warped image with the same shape of img
    """
    warnings.warn('This function is just for prototyping and cannot '
                  'guarantee the computational efficiency.')
    assert flow.ndim == 3, 'Flow must be in 3D arrays.'
    height = flow.shape[0]
    width = flow.shape[1]
    channels = img.shape[2]

    output = np.ones(
        (height, width, channels), dtype=img.dtype) * filling_value

    grid = np.indices((height, width)).swapaxes(0, 1).swapaxes(1, 2)
    dx = grid[:, :, 0] + flow[:, :, 1]
    dy = grid[:, :, 1] + flow[:, :, 0]
    sx = np.floor(dx).astype(int)
    sy = np.floor(dy).astype(int)
    valid = (sx >= 0) & (sx < height - 1) & (sy >= 0) & (sy < width - 1)

    if interpolate_mode == 'nearest':
        output[valid, :] = img[dx[valid].round().astype(int),
                               dy[valid].round().astype(int), :]
    elif interpolate_mode == 'bilinear':
        # dirty walkround for integer positions
        eps_ = 1e-6
        dx, dy = dx + eps_, dy + eps_
        left_top_ = img[np.floor(dx[valid]).astype(int),
                        np.floor(dy[valid]).astype(int), :] * (
                            np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                np.ceil(dy[valid]) - dy[valid])[:, None]
        left_down_ = img[np.ceil(dx[valid]).astype(int),
                         np.floor(dy[valid]).astype(int), :] * (
                             dx[valid] - np.floor(dx[valid]))[:, None] * (
                                 np.ceil(dy[valid]) - dy[valid])[:, None]
        right_top_ = img[np.floor(dx[valid]).astype(int),
                         np.ceil(dy[valid]).astype(int), :] * (
                             np.ceil(dx[valid]) - dx[valid])[:, None] * (
                                 dy[valid] - np.floor(dy[valid]))[:, None]
        right_down_ = img[np.ceil(dx[valid]).astype(int),
                          np.ceil(dy[valid]).astype(int), :] * (
                              dx[valid] - np.floor(dx[valid]))[:, None] * (
                                  dy[valid] - np.floor(dy[valid]))[:, None]
        output[valid, :] = left_top_ + left_down_ + right_top_ + right_down_
    else:
        raise NotImplementedError(
            'We only support interpolation modes of nearest and bilinear, '
            f'but got {interpolate_mode}.')
    return output.astype(img.dtype)


def flow_from_bytes(content: bytes) -> np.ndarray:
    """Read dense optical flow from bytes.

    .. note::
        This load optical flow function works for FlyingChairs, FlyingThings3D,
        Sintel, FlyingChairsOcc datasets, but cannot load the data from
        ChairsSDHom.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        ndarray: Loaded optical flow with the shape (H, W, 2).
    """

    # header in first 4 bytes
    header = content[:4]
    if header.decode('utf-8') != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')
    # width in second 4 bytes
    width = np.frombuffer(content[4:], np.int32, 1).squeeze()
    # height in third 4 bytes
    height = np.frombuffer(content[8:], np.int32, 1).squeeze()
    # after first 12 bytes, all bytes are flow
    flow = np.frombuffer(content[12:], np.float32, width * height * 2).reshape(
        (height, width, 2))

    return flow


def sparse_flow_from_bytes(content: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Read the optical flow in KITTI datasets from bytes.

    This function is modified from RAFT load the `KITTI datasets
    <https://github.com/princeton-vl/RAFT/blob/224320502d66c356d88e6c712f38129e60661e80/core/utils/frame_utils.py#L102>`_.

    Args:
        content (bytes): Optical flow bytes got from files or other streams.

    Returns:
        Tuple(ndarray, ndarray): Loaded optical flow with the shape (H, W, 2)
        and flow valid mask with the shape (H, W).
    """  # nopa

    content = np.frombuffer(content, np.uint8)
    flow = cv2.imdecode(content, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid
