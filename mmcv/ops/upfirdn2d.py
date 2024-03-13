# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# source: https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/upfirdn2d.py # noqa
"""Custom PyTorch ops for efficient resampling of 2D images."""
from typing import Dict, List, Union

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


def _get_filter_size(filter):
    """get width and height of filter kernel."""
    if filter is None:
        return 1, 1
    assert isinstance(filter, torch.Tensor) and filter.ndim in [1, 2]
    fw = filter.shape[-1]
    fh = filter.shape[0]
    fw = int(fw)
    fh = int(fh)
    assert fw >= 1 and fh >= 1
    return fw, fh


def upfirdn2d(input: torch.Tensor,
              filter: torch.Tensor,
              up: int = 1,
              down: int = 1,
              padding: Union[int, List[int]] = 0,
              flip_filter: bool = False,
              gain: Union[float, int] = 1,
              use_custom_op: bool = True):
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
        filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
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
        use_custom_op (bool): Whether to use customized op.
            Defaults to True.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`
    """
    assert isinstance(input, torch.Tensor)
    if use_custom_op and input.device.type == 'cuda':
        return _upfirdn2d_cuda(
            up=up,
            down=down,
            padding=padding,
            flip_filter=flip_filter,
            gain=gain).apply(input, filter)
    elif use_custom_op and input.device.type == 'musa':
        return _upfirdn2d_musa(
            up=up,
            down=down,
            padding=padding,
            flip_filter=flip_filter,
            gain=gain).apply(input, filter)
    return _upfirdn2d_ref(
        input,
        filter,
        up=up,
        down=down,
        padding=padding,
        flip_filter=flip_filter,
        gain=gain)


def _upfirdn2d_ref(input: torch.Tensor,
                   filter: torch.Tensor,
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
        filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
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
    if filter is None:
        filter = torch.ones([1, 1], dtype=torch.float32, device=input.device)
    assert isinstance(filter, torch.Tensor) and filter.ndim in [1, 2]
    assert filter.dtype == torch.float32 and not filter.requires_grad
    batch_size, num_channels, in_height, in_width = input.shape
    upx, upy = _parse_scaling(up)
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Check that upsampled buffer is not smaller than the filter.
    upW = in_width * upx + padx0 + padx1
    upH = in_height * upy + pady0 + pady1
    assert upW >= filter.shape[-1] and upH >= filter.shape[0]

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
    filter = filter * (gain**(filter.ndim / 2))
    filter = filter.to(x.dtype)
    if not flip_filter:
        filter = filter.flip(list(range(filter.ndim)))

    # Convolve with the filter.
    filter = filter[None, None].repeat([num_channels, 1] + [1] * filter.ndim)
    if filter.ndim == 4:
        x = conv2d(input=x, weight=filter, groups=num_channels)
    else:
        x = conv2d(input=x, weight=filter.unsqueeze(2), groups=num_channels)
        x = conv2d(input=x, weight=filter.unsqueeze(3), groups=num_channels)

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


_upfirdn2d_musa_cache: Dict = dict()


def _upfirdn2d_musa(up: int = 1,
                    down: int = 1,
                    padding: Union[int, List[int]] = 0,
                    flip_filter: bool = False,
                    gain: Union[float, int] = 1):
    """Fast MUSA implementation of `upfirdn2d()` using custom ops.

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
    if key in _upfirdn2d_musa_cache:
        return _upfirdn2d_musa_cache[key]

    # Forward op.
    class Upfirdn2dMusa(torch.autograd.Function):

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
                dx = _upfirdn2d_musa(
                    up=down,
                    down=up,
                    padding=p,
                    flip_filter=(not flip_filter),
                    gain=gain).apply(dy, f)

            assert not ctx.needs_input_grad[1]
            return dx, df

    # Add to cache.
    _upfirdn2d_musa_cache[key] = Upfirdn2dMusa
    return Upfirdn2dMusa


def filter2d(input: torch.Tensor,
             filter: torch.Tensor,
             padding: Union[int, List[int]] = 0,
             flip_filter: bool = False,
             gain: Union[float, int] = 1,
             use_custom_op: bool = True):
    """Filter a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape matches the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
            filter_width]` (non-separable), `[filter_taps]` (separable), or
            `None`.
        padding (int | tuple[int]): Padding with respect to the output.
            Can be a single number or a list/tuple `[x, y]` or `[x_before,
            x_after, y_before, y_after]`. Defaults to 0.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        gain (int): Overall scaling factor for signal magnitude.
            Defaults to 1.
        use_custom_op (bool): Whether to use customized op.
            Defaults to True.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
        out_width]`.
    """
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(filter)
    p = [
        padx0 + fw // 2,
        padx1 + (fw - 1) // 2,
        pady0 + fh // 2,
        pady1 + (fh - 1) // 2,
    ]
    return upfirdn2d(
        input,
        filter,
        padding=p,
        flip_filter=flip_filter,
        gain=gain,
        use_custom_op=use_custom_op)


def upsample2d(input: torch.Tensor,
               filter: torch.Tensor,
               up: int = 2,
               padding: Union[int, List[int]] = 0,
               flip_filter: bool = False,
               gain: Union[float, int] = 1,
               use_custom_op: bool = True):
    """Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
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
        use_custom_op (bool): Whether to use customized op.
            Defaults to True.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
        out_height, out_width]`
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(filter)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(
        input,
        filter,
        up=up,
        padding=p,
        flip_filter=flip_filter,
        gain=gain * upx * upy,
        use_custom_op=use_custom_op)


def downsample2d(input: torch.Tensor,
                 filter: torch.Tensor,
                 down: int = 2,
                 padding: Union[int, List[int]] = 0,
                 flip_filter: bool = False,
                 gain: Union[float, int] = 1,
                 use_custom_op: bool = True):
    """Downsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a fraction of the
    input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        input (torch.Tensor): Float32/float64/float16 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        filter (torch.Tensor): Float32 FIR filter of the shape `[filter_height,
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
        use_custom_op (bool): Whether to use customized op.
            Defaults to True.

    Returns:
        torch.Tensor: Tensor of the shape `[batch_size, num_channels,
        out_height, out_width]`.
    """
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(filter)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(
        input,
        filter,
        down=down,
        padding=p,
        flip_filter=flip_filter,
        gain=gain,
        use_custom_op=use_custom_op)
