# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# source: https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/upfirdn2d.py # noqa
"""Custom PyTorch ops for efficient resampling of 2D images."""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ..utils import ext_loader
from .conv2d_gradfix import conv2d

ext_module = ext_loader.load_ext('_ext', ['upfirdn2d'])


def _parse_scaling(scaling):
    """parse scaling into list [x, y]"""
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy


def _parse_padding(padding):
    """parse padding into list [padx0, padx1, pady0, pady1]"""
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _get_filter_size(f):
    """get width and height of filter kernel."""
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    fw = f.shape[-1]
    fh = f.shape[0]
    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh


def setup_filter(f: torch.Tensor,
                 device: Any = torch.device('cpu'),
                 normalize: bool = True,
                 flip_filter: bool = False,
                 gain: Union[float, int] = 1,
                 separable: Optional[bool] = None):
    """Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        f (torch.Tensor): Torch tensor, numpy array, or python list
            of the shape `[filter_height, filter_width]`
            (non-separable), `[filter_taps]` (separable), `[]`
            (impulse), or `None`.
        device (Any): Result device. Defaults to torch.device('cpu').
        normalize (bool): Normalize the filter so that it retains the
            magnitude for constant input signal (DC). Defaults to True.
        flip_filter (bool): Flip the filter? Defaults to False.
        gain (Union[float, int]): Overall scaling factor for signal
            magnitude. Defaults to 1.
        separable (Optional[bool]): Return a separable filter?
            Defaults to None.

    Returns:
        torch.Tensor: Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if f is None:
        f = 1
    f = torch.as_tensor(f, dtype=torch.float32)
    assert f.ndim in [0, 1, 2]
    assert f.numel() > 0
    if f.ndim == 0:
        f = f[np.newaxis]

    # Separable?
    if separable is None:
        separable = (f.ndim == 1 and f.numel() >= 8)
    if f.ndim == 1 and not separable:
        f = f.outer(f)
    assert f.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        f /= f.sum()
    if flip_filter:
        f = f.flip(list(range(f.ndim)))
    f = f * (gain**(f.ndim / 2))
    f = f.to(device=device)
    return f


def upfirdn2d(input: torch.Tensor,
              f: torch.Tensor,
              up: int = 1,
              down: int = 1,
              padding: Union[int, List[int]] = 0,
              flip_filter: bool = False,
              gain: Union[float, int] = 1,
              impl: str = 'cuda'):
    """Pad, upsample, filter, and downsample a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    2. Pad the image with the specified number of zeros on each side
    (`padding`). Negative padding corresponds to cropping the image.

    3. Convolve the image with the specified 2D FIR filter (`f`),
    shrinking it so that the footprint of all output pixels lies within
    the input image.

    4. Downsample the image by keeping every Nth pixel (`down`).

    This sequence of operations bears close resemblance to
        scipy.signal.upfirdn().

    The fused op is considerably more efficient than performing the same
    calculation using standard PyTorch ops. It supports gradients of arbitrary
    order.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a
            list/tuple `[x, y]`. Defaults to 1.
        down (int): Integer downsampling factor. Can be a single int
            or a list/tuple `[x, y]`. Defaults to 1.
        padding (int | tuple[int]): Padding with respect to the upsampled
            image. Can be a single number or a list/tuple `[x, y]` or
            `[x_before, x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        gain (int): Overall scaling factor for signal magnitude.
            Defaults to 1.
        impl (str): Implementation to use. Can be `'ref'` or
            `'cuda'`. If set to `'cuda'`, fast CUDA implementation of
            `upfirdn2d()` using custom ops will be used. If set to `'ref'`,
            slow reference implementation of `upfirdn2d()` using standard
            PyTorch ops will be used. Defaults to 'cuda'.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`
    """
    assert isinstance(input, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and input.device.type == 'cuda':
        return _upfirdn2d_cuda(
            up=up,
            down=down,
            padding=padding,
            flip_filter=flip_filter,
            gain=gain).apply(input, f)
    return _upfirdn2d_ref(
        input,
        f,
        up=up,
        down=down,
        padding=padding,
        flip_filter=flip_filter,
        gain=gain)


def _upfirdn2d_ref(input: torch.Tensor,
                   f: torch.Tensor,
                   up: int = 1,
                   down: int = 1,
                   padding: Union[int, List[int]] = 0,
                   flip_filter: bool = False,
                   gain: Union[float, int] = 1):
    """Slow reference implementation of `upfirdn2d()` using standard PyTorch
    ops.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a
            list/tuple `[x, y]`. Defaults to 1.
        down (int): Integer downsampling factor. Can be a single int
            or a list/tuple `[x, y]`. Defaults to 1.
        padding (int | tuple[int]): Padding with respect to the upsampled
            image. Can be a single number or a list/tuple `[x, y]` or
            `[x_before, x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        gain (int): Overall scaling factor for signal magnitude.
            Defaults to 1.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
            out_height, out_width]`.
    """
    # Validate arguments.
    assert isinstance(input, torch.Tensor) and input.ndim == 4
    if f is None:
        f = torch.ones([1, 1], dtype=torch.float32, device=input.device)
    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = input.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= f.shape[-1] and upH >= f.shape[0]

    # Upsample by inserting zeros.
    x = input.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = torch.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = torch.nn.functional.pad(
        x, [max(padx0, 0),
            max(padx1, 0),
            max(pady0, 0),
            max(pady1, 0)])
    x = x[:, :,
          max(-pady0, 0):x.shape[2] - max(-pady1, 0),
          max(-padx0, 0):x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    f = f * (gain**(f.ndim / 2))
    f = f.to(x.dtype)
    if not flip_filter:
        f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    if f.ndim == 4:
        x = conv2d(input=x, weight=f, groups=num_channels)
    else:
        x = conv2d(input=x, weight=f.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=f.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::downy, ::downx]
    return x


_upfirdn2d_cuda_cache: Dict = dict()


def _upfirdn2d_cuda(up: int = 1,
                    down: int = 1,
                    padding: Union[int, List[int]] = 0,
                    flip_filter: bool = False,
                    gain: Union[float, int] = 1):
    """Fast CUDA implementation of `upfirdn2d()` using custom ops.

    Args:
        up (int): Integer upsampling factor. Can be a single int or a
            list/tuple `[x, y]`. Defaults to 1.
        down (int): Integer downsampling factor. Can be a single int
            or a list/tuple `[x, y]`. Defaults to 1.
        padding (int | tuple[int]): Padding with respect to the upsampled
            image. Can be a single number or a list/tuple `[x, y]` or
            `[x_before, x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        gain (int): Overall scaling factor for signal magnitude.
            Defaults to 1.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
            out_height, out_width]`
    """
    # Parse arguments.
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Lookup from cache.
    key = (upx, upy, downx, downy, padx0, padx1, pady0, pady1, flip_filter,
           gain)
    if key in _upfirdn2d_cuda_cache:
        return _upfirdn2d_cuda_cache[key]

    # Forward op.
    class Upfirdn2dCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, f):  # pylint: disable=arguments-differ
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if f is None:
                f = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if f.ndim == 1 and f.shape[0] == 1:
                f = f.square().unsqueeze(
                    0)  # Convert separable-1 into full-1x1.
            assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
            y = x
            if f.ndim == 2:
                y = ext_module.upfirdn2d(y, f, upx, upy, downx, downy, padx0,
                                         padx1, pady0, pady1, flip_filter,
                                         gain)
            else:
                y = ext_module.upfirdn2d(y, f.unsqueeze(0), upx, 1, downx, 1,
                                         padx0, padx1, 0, 0, flip_filter, 1.0)
                y = ext_module.upfirdn2d(y, f.unsqueeze(1), 1, upy, 1, downy,
                                         0, 0, pady0, pady1, flip_filter, gain)
            ctx.save_for_backward(f)
            ctx.x_shape = x.shape
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            f, = ctx.saved_tensors
            _, _, ih, iw = ctx.x_shape
            _, _, oh, ow = dy.shape
            fw, fh = _get_filter_size(f)
            p = [
                fw - padx0 - 1,
                iw * upx - ow * downx + padx0 - upx + 1,
                fh - pady0 - 1,
                ih * upy - oh * downy + pady0 - upy + 1,
            ]
            dx = None
            df = None

            if ctx.needs_input_grad[0]:
                dx = _upfirdn2d_cuda(
                    up=down,
                    down=up,
                    padding=p,
                    flip_filter=(not flip_filter),
                    gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_cuda_cache[key] = Upfirdn2dCuda
    return Upfirdn2dCuda


def filter2d(input: torch.Tensor,
             f: torch.Tensor,
             padding: Union[int, List[int]] = 0,
             flip_filter: bool = False,
             gain: Union[float, int] = 1,
             impl: str = 'cuda'):
    """Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None`.
        padding (int | tuple[int]): Padding with respect to the output.
            Can be a single number or a list/tuple `[x, y]` or `[x_before,
            x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        gain (int): Overall scaling factor for signal magnitude.
            Defaults to 1.
        impl (str): Implementation to use for `upfirdn2d`. Defaults to 'cuda'.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
            out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(
        input, f, padding=p, flip_filter=flip_filter, gain=gain, impl=impl)


def upsample2d(input: torch.Tensor,
               f: torch.Tensor,
               up: int = 2,
               padding: Union[int, List[int]] = 0,
               flip_filter: bool = False,
               gain: Union[float, int] = 1,
               impl: str = 'cuda'):
    """Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None` (identity).
        up (int): Integer upsampling factor. Can be a single int or a
            list/tuple `[x, y]`. Defaults to 2.
        padding (int | tuple[int]): Padding with respect to the output.
            Can be a single number or a list/tuple `[x, y]` or `[x_before,
            x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation. Defaults
            to False.
        gain (int): Overall scaling factor for signal magnitude. Defaults to 1.
        impl (str): Implementation to use for `upfirdn2d`. Defaults to 'cuda'.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
            out_height, out_width]`
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(
        input,
        f,
        up=up,
        padding=p,
        flip_filter=flip_filter,
        gain=gain * upx * upy,
        impl=impl)


def downsample2d(input: torch.Tensor,
                 f: torch.Tensor,
                 down: int = 2,
                 padding: Union[int, List[int]] = 0,
                 flip_filter: bool = False,
                 gain: Union[float, int] = 1,
                 impl: str = 'cuda'):
    """Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        f (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None` (identity).
        down (int): Integer downsampling factor. Can be a single int or a
                     list/tuple `[x, y]` (default: 1). Defaults to 2.
        padding (int | tuple[int]): Padding with respect to the input.
            Can be a single number or a list/tuple `[x, y]` or `[x_before,
            x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation. Defaults
            to False.
        gain (int): Overall scaling factor for signal magnitude. Defaults to 1.
        impl (str): Implementation to use for `upfirdn2d`. Defaults to 'cuda'.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
            out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(
        input,
        f,
        down=down,
        padding=p,
        flip_filter=flip_filter,
        gain=gain,
        impl=impl)
