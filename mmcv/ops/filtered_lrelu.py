# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# source: https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/filtered_lrelu.py # noqa
import warnings
from typing import Dict, Optional, Union

import numpy as np
import torch

from ..utils import ext_loader
from .bias_act import bias_act
from .upfirdn2d import upfirdn2d

ext_module = ext_loader.load_ext('_ext',
                                 ['filtered_lrelu', 'filtered_lrelu_act_'])

_plugin = None


def _get_filter_size(f):
    if f is None:
        return 1, 1
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return f.shape[-1], f.shape[0]  # width, height


def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, (int, np.integer)) for x in padding)
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        px, py = padding
        padding = [px, px, py, py]
    px0, px1, py0, py1 = padding
    return px0, px1, py0, py1


def filtered_lrelu(input: torch.Tensor,
                   fu: Optional[torch.Tensor] = None,
                   fd: Optional[torch.Tensor] = None,
                   bias: Optional[torch.Tensor] = None,
                   up: int = 1,
                   down: int = 1,
                   padding: int = 0,
                   gain: float = np.sqrt(2),
                   slope: float = 0.2,
                   clamp: Optional[Union[float, int]] = None,
                   flip_filter: bool = False,
                   use_custom_op: bool = True):
    """Filtered leaky ReLU for a batch of 2D images.

    Performs the following sequence of operations for each channel:

    1. Add channel-specific bias if provided (`b`).

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side
      (`padding`). Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter (`fu`),
       shrinking it so that the footprint of all output pixels lies within the
       input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is
    provided.

    8. Convolve the image with the specified downsampling FIR filter (`fd`),
       shrinking it so that the footprint of all output pixels lies within the
       input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same
    calculation using standard PyTorch ops. It supports gradients of arbitrary
    order.

    Args:
        input (torch.Tensor): Float32/float16/float64 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        fu (torch.Tensor): Float32 upsampling FIR filter of the shape
            `[filter_height, filter_width]` (non-separable), `[filter_taps]`
            (separable), or `None` (identity). Defaults to None.
        fd (torch.Tensor): Float32 downsampling FIR filter of the
            shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
            Defaults to None.
        bias (torch.Tensor): Bias vector, or `None` to disable. Must be
            a 1D tensor of the same type as `x`. The length of vector must must
            match the channel dimension of `x`. Defaults to None.
        up (int): Integer upsampling factor. Defaults to 1.
        down (int): Integer downsampling factor. Defaults to 1.
        padding (int): Padding with respect to the upsampled image. Can be a
            single number or a list/tuple `[x, y]` or `[x_before, x_after,
            y_before, y_after]`. Defaults to 0.
        gain (float): Overall scaling factor for signal magnitude.
            Defaults to np.sqrt(2).
        slope (float): Slope on the negative side of leaky ReLU.
            Defaults to 0.2.
        clamp (Optional[Union[float, int]]): Maximum magnitude for leaky ReLU
            output. Defaults to None.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.
        use_custom_op (bool): Whether to use customized op.
            Defaults to True.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
                out_width]`.
    """
    assert isinstance(input, torch.Tensor)
    if use_custom_op and input.is_cuda:
        return _filtered_lrelu_cuda(
            up=up,
            down=down,
            padding=padding,
            gain=gain,
            slope=slope,
            clamp=clamp,
            flip_filter=flip_filter).apply(input, fu, fd, bias, None, 0, 0)
    return _filtered_lrelu_ref(
        input,
        fu=fu,
        fd=fd,
        bias=bias,
        up=up,
        down=down,
        padding=padding,
        gain=gain,
        slope=slope,
        clamp=clamp,
        flip_filter=flip_filter)


def _filtered_lrelu_ref(input: torch.Tensor,
                        fu: Optional[torch.Tensor] = None,
                        fd: Optional[torch.Tensor] = None,
                        bias: Optional[torch.Tensor] = None,
                        up: int = 1,
                        down: int = 1,
                        padding: int = 0,
                        gain: float = np.sqrt(2),
                        slope: float = 0.2,
                        clamp: Optional[Union[float, int]] = None,
                        flip_filter: bool = False):
    """Slow and memory-inefficient reference implementation of
    `filtered_lrelu()` using existing `upfirdn2n()` and `bias_act()` ops.

    Args:
        input (torch.Tensor): Float32/float16/float64 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        fu (torch.Tensor): Float32 upsampling FIR filter of the shape
            `[filter_height, filter_width]` (non-separable), `[filter_taps]`
            (separable), or `None` (identity). Defaults to None.
        fd (torch.Tensor): Float32 downsampling FIR filter of the
            shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
            Defaults to None.
        bias (torch.Tensor): Bias vector, or `None` to disable. Must be
            a 1D tensor of the same type as `x`. The length of vector must must
            match the channel dimension of `x`. Defaults to None.
        up (int): Integer upsampling factor. Defaults to 1.
        down (int): Integer downsampling factor. Defaults to 1.
        padding (int): Padding with respect to the upsampled image. Can be a
            single number or a list/tuple `[x, y]` or `[x_before, x_after,
            y_before, y_after]`. Defaults to 0.
        gain (float): Overall scaling factor for signal magnitude.
            Defaults to np.sqrt(2).
        slope (float): Slope on the negative side of leaky ReLU.
            Defaults to 0.2.
        clamp (float | int): Maximum magnitude for leaky ReLU
            output. Defaults to None.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
                out_width]`.
    """
    assert isinstance(input, torch.Tensor) and input.ndim == 4
    fu_w, fu_h = _get_filter_size(fu)
    fd_w, fd_h = _get_filter_size(fd)
    if bias is not None:
        assert isinstance(bias, torch.Tensor) and bias.dtype == input.dtype
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)

    # Calculate output size.
    batch_size, channels, in_h, in_w = input.shape
    in_dtype = input.dtype
    out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) +
             (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) +
             (down - 1)) // down

    # Compute using existing ops.
    x = bias_act(input=input, bias=bias)  # Apply bias.
    x = upfirdn2d(
        input=x,
        f=fu,
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter)  # Upsample.
    x = bias_act(
        input=x, act='lrelu', alpha=slope, gain=gain,
        clamp=clamp)  # Bias, leaky ReLU, clamp.
    x = upfirdn2d(
        input=x, f=fd, down=down, flip_filter=flip_filter)  # Downsample.

    assert x.shape == (batch_size, channels, out_h, out_w)
    assert x.dtype == in_dtype
    return x


_filtered_lrelu_cuda_cache: Dict = dict()


def _filtered_lrelu_cuda(up: int = 1,
                         down: int = 1,
                         padding: int = 0,
                         gain: float = np.sqrt(2),
                         slope: float = 0.2,
                         clamp: Optional[Union[float, int]] = None,
                         flip_filter: bool = False):
    """Fast CUDA implementation of `filtered_lrelu()` using custom ops.

    Args:
        up (int): Integer upsampling factor. Defaults to 1.
        down (int): Integer downsampling factor. Defaults to 1.
        padding (int): Padding with respect to the upsampled image. Can be a
            single number or a list/tuple `[x, y]` or `[x_before, x_after,
            y_before, y_after]`. Defaults to 0.
        gain (float): Overall scaling factor for signal magnitude.
            Defaults to np.sqrt(2).
        slope (float): Slope on the negative side of leaky ReLU.
            Defaults to 0.2.
        clamp (float | int): Maximum magnitude for leaky ReLU
            output. Defaults to None.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
                out_width]`.
    """
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    px0, px1, py0, py1 = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')

    # Lookup from cache.
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    # Forward op.
    class FilteredLReluCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, fu, fd, bias, si, sx, sy):
            # pylint: disable=arguments-differ
            assert isinstance(input, torch.Tensor) and input.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels
            # (faster than separable).
            if fu is None:
                fu = torch.ones([1, 1],
                                dtype=torch.float32,
                                device=input.device)
            if fd is None:
                fd = torch.ones([1, 1],
                                dtype=torch.float32,
                                device=input.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale
            # factor is 1.
            if up == 1 and fu.ndim == 1 and fu.shape[0] == 1:
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and fd.shape[0] == 1:
                fd = fd.square()[None]

            # Missing sign input tensor.
            if si is None:
                si = torch.empty([0])

            # Missing bias tensor.
            if bias is None:
                bias = torch.zeros([input.shape[1]],
                                   dtype=input.dtype,
                                   device=input.device)

            # Construct internal sign tensor only if gradients are needed.
            write_signs = (si.numel() == 0) and (input.requires_grad
                                                 or bias.requires_grad)

            # Warn if input storage strides are not in decreasing order due to
            # e.g. channels-last layout.
            strides = [
                input.stride(i) for i in range(input.ndim) if input.size(i) > 1
            ]
            if any(a < b for a, b in zip(strides[:-1], strides[1:])):
                warnings.warn(
                    'low-performance memory layout detected in filtered_lrelu '
                    'input', RuntimeWarning)

            # Call C++/Cuda plugin if datatype is supported.
            if input.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(
                        input.device) != torch.cuda.default_stream(
                            input.device):
                    warnings.warn(
                        'filtered_lrelu called with non-default cuda stream '
                        'but concurrent execution is not supported',
                        RuntimeWarning)
                y, so, return_code = ext_module.filtered_lrelu(
                    input, fu, fd, bias, si.to(input.device), up, down, px0,
                    px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter,
                    write_signs)
            else:
                return_code = -1

            # No Cuda kernel found? Fall back to generic implementation.
            # Still more memory efficient than the reference implementation
            # because only the bit-packed sign tensor is retained for gradient
            # computation.
            if return_code < 0:
                warnings.warn(
                    'filtered_lrelu called with parameters that have no '
                    'optimized CUDA kernel, using generic fallback',
                    RuntimeWarning)

                y = input.add(bias.unsqueeze(-1).unsqueeze(-1))  # Add bias.
                y = upfirdn2d(
                    input=y,
                    f=fu,
                    up=up,
                    padding=[px0, px1, py0, py1],
                    gain=float(up**2),
                    flip_filter=flip_filter)  # Upsample.
                # Activation function and sign handling. Modifies y in-place.
                so = ext_module.filtered_lrelu_act_(y, si.to(y.device), sx, sy,
                                                    gain, slope, clamp,
                                                    write_signs)
                y = upfirdn2d(
                    input=y, f=fd, down=down,
                    flip_filter=flip_filter)  # Downsample.

            # Prepare for gradient computation.
            ctx.save_for_backward(fu, fd, (si if si.numel() else so))
            ctx.x_shape = input.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            fu, fd, si = ctx.saved_tensors
            _, _, xh, xw = ctx.x_shape
            _, _, yh, yw = ctx.y_shape
            sx, sy = ctx.s_ofs
            dx = None  # 0
            dfu = None
            assert not ctx.needs_input_grad[1]
            dfd = None
            assert not ctx.needs_input_grad[2]
            db = None  # 3
            dsi = None
            assert not ctx.needs_input_grad[4]
            dsx = None
            assert not ctx.needs_input_grad[5]
            dsy = None
            assert not ctx.needs_input_grad[6]

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [
                    (fu.shape[-1] - 1) + (fd.shape[-1] - 1) - px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (fu.shape[0] - 1) + (fd.shape[0] - 1) - py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up**2) / (down**2)
                ff = (not flip_filter)
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(
                    up=down,
                    down=up,
                    padding=pp,
                    gain=gg,
                    slope=slope,
                    clamp=None,
                    flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda