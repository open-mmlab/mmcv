# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np

from mmcv._flow_warp_ext import flow_warp_c
from mmcv.arraymisc import dequantize, quantize
from mmcv.image import imread, imwrite
from mmcv.utils import is_str


def flowread(flow_or_path, quantize=False, concat_axis=0, *args, **kwargs):
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
                raise IOError(f'Invalid flow file: {flow_or_path}')
            else:
                if header != 'PIEH':
                    raise IOError(f'Invalid flow file: {flow_or_path}, '
                                  'header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
    else:
        assert concat_axis in [0, 1]
        cat_flow = imread(flow_or_path, flag='unchanged')
        if cat_flow.ndim != 2:
            raise IOError(
                f'{flow_or_path} is not a valid quantized flow file, '
                f'its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)

    return flow.astype(np.float32)


def flowwrite(flow, filename, quantize=False, concat_axis=0, *args, **kwargs):
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
            f.write('PIEH'.encode('utf-8'))
            np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
            flow = flow.astype(np.float32)
            flow.tofile(f)
            f.flush()
    else:
        assert concat_axis in [0, 1]
        dx, dy = quantize_flow(flow, *args, **kwargs)
        dxdy = np.concatenate((dx, dy), axis=concat_axis)
        imwrite(dxdy, filename)


def quantize_flow(flow, max_val=0.02, norm=True):
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


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
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

    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def flow_warp(img, flow, filling_value=0, interpolate_mode='nearest'):
    """Use flow to warp img.

    Args:
        img (ndarray, float or uint8): Image to be warped.
        flow (ndarray, float): Optical Flow.
        filling_value (int): The missing pixels will be set with filling_value.
        interpolate_mode (str): bilinear -> Bilinear Interpolation;
                                nearest -> Nearest Neighbor.

    Returns:
        ndarray: Warped image with the same shape of img
    """
    interpolate_mode_dict = {'bilinear': 0, 'nearest': 1}
    assert len(img.shape) == 3
    assert len(flow.shape) == 3 and flow.shape[2] == 2
    assert flow.shape[:2] == img.shape[:2]
    assert interpolate_mode in interpolate_mode_dict.keys()

    interpolate_mode = interpolate_mode_dict[interpolate_mode]
    img_float = img.astype(np.float64)

    out = flow_warp_c(
        img_float,
        flow.astype(np.float64),
        filling_value=filling_value,
        interpolate_mode=interpolate_mode)

    return out
