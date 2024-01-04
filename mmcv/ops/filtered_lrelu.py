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
from .upfirdn2d import _get_filter_size, _parse_padding, upfirdn2d

ext_module = ext_loader.load_ext('_ext',
                                 ['filtered_lrelu', 'filtered_lrelu_act_'])

_plugin = None


def filtered_lrelu(input: torch.Tensor,
                   filter_up: Optional[torch.Tensor] = None,
                   filter_down: Optional[torch.Tensor] = None,
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

    1. Add channel-specific bias if `bias` is provided.

    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).

    3. Pad the image with the specified number of zeros on each side
      (`padding`). Negative padding corresponds to cropping the image.

    4. Convolve the image with the specified upsampling FIR filter
        (`filter_up`), shrinking it so that the footprint of all output pixels
        lies within the input image.

    5. Multiply each value by the provided gain factor (`gain`).

    6. Apply leaky ReLU activation function to each value.

    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is
       provided.

    8. Convolve the image with the specified downsampling FIR filter
        (`filter_down`), shrinking it so that the footprint of all output
        pixels lies within the input image.

    9. Downsample the image by keeping every Nth pixel (`down`).

    The fused op is considerably more efficient than performing the same
    calculation using standard PyTorch ops. It supports gradients of arbitrary
    order.

    Args:
        input (torch.Tensor): Float32/float16/float64 input tensor of the shape
            `[batch_size, num_channels, in_height, in_width]`.
        filter_up (torch.Tensor): Float32 upsampling FIR filter of the shape
            `[filter_height, filter_width]` (non-separable), `[filter_taps]`
            (separable), or `None` (identity). Defaults to None.
        filter_down (torch.Tensor): Float32 downsampling FIR filter of the
            shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
            Defaults to None.
        bias (torch.Tensor): Bias vector, or `None` to disable. Must be
            a 1D tensor of the same type as `input`. The length of vector must
            match the channel dimension of `input`. Defaults to None.
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
            flip_filter=flip_filter).apply(input, filter_up, filter_down, bias,
                                           None, 0, 0)
    if use_custom_op and input.is_musa:
        return _filtered_lrelu_musa(
            up=up,
            down=down,
            padding=padding,
            gain=gain,
            slope=slope,
            clamp=clamp,
            flip_filter=flip_filter).apply(input, filter_up, filter_down, bias,
                                           None, 0, 0)
    return _filtered_lrelu_ref(
        input,
        filter_up=filter_up,
        filter_down=filter_down,
        bias=bias,
        up=up,
        down=down,
        padding=padding,
        gain=gain,
        slope=slope,
        clamp=clamp,
        flip_filter=flip_filter)


def _filtered_lrelu_ref(input: torch.Tensor,
                        filter_up: Optional[torch.Tensor] = None,
                        filter_down: Optional[torch.Tensor] = None,
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
        filter_up (torch.Tensor): Float32 upsampling FIR filter of the shape
            `[filter_height, filter_width]` (non-separable), `[filter_taps]`
            (separable), or `None` (identity). Defaults to None.
        filter_down (torch.Tensor): Float32 downsampling FIR filter of the
            shape `[filter_height, filter_width]` (non-separable),
            `[filter_taps]` (separable), or `None` (identity).
            Defaults to None.
        bias (torch.Tensor): Bias vector, or `None` to disable. Must be
            a 1D tensor of the same type as `input`. The length of vector must
            match the channel dimension of `input`. Defaults to None.
        up (int): Integer upsampling factor. Defaults to 1.
        down (int): Integer downsampling factor. Defaults to 1.
        padding (int): Padding with respect to the upsampled image. Can be a
            single number or a list/tuple `[x, y]` or `[x_before, x_after,
            y_before, y_after]`. Defaults to 0.
        gain (float): Overall scaling factor for signal magnitude.
            Defaults to np.sqrt(2).
        slope (float): Slope on the negative side of leaky ReLU.
            Defaults to 0.2.
        clamp (float or int): Maximum magnitude for leaky ReLU
            output. Defaults to None.
        flip_filter (bool): False = convolution, True = correlation.
            Defaults to False.

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height,
        out_width]`.
    """
    assert isinstance(input, torch.Tensor) and input.ndim == 4
    filter_up_w, filter_up_h = _get_filter_size(filter_up)
    filter_down_w, filter_down_h = _get_filter_size(filter_down)
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
    out_w = (in_w * up + (px0 + px1) - (filter_up_w - 1) -
             (filter_down_w - 1) + (down - 1)) // down
    out_h = (in_h * up + (py0 + py1) - (filter_up_h - 1) -
             (filter_down_h - 1) + (down - 1)) // down

    # Compute using existing ops.
    output = bias_act(input=input, bias=bias)  # Apply bias.
    output = upfirdn2d(
        input=output,
        filter=filter_up,
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
        flip_filter=flip_filter)  # Upsample.
    output = bias_act(
        input=output, act='lrelu', alpha=slope, gain=gain,
        clamp=clamp)  # Bias, leaky ReLU, clamp.
    output = upfirdn2d(
        input=output, filter=filter_down, down=down,
        flip_filter=flip_filter)  # Downsample.

    assert output.shape == (batch_size, channels, out_h, out_w)
    assert output.dtype == in_dtype
    return output


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
        clamp (float or int): Maximum magnitude for leaky ReLU
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
        def forward(ctx, input, filter_up, filter_down, bias, si, sx, sy):
            # pylint: disable=arguments-differ
            assert isinstance(input, torch.Tensor) and input.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels
            # (faster than separable).
            if filter_up is None:
                filter_up = torch.ones([1, 1],
                                       dtype=torch.float32,
                                       device=input.device)
            if filter_down is None:
                filter_down = torch.ones([1, 1],
                                         dtype=torch.float32,
                                         device=input.device)
            assert 1 <= filter_up.ndim <= 2
            assert 1 <= filter_down.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale
            # factor is 1.
            if up == 1 and filter_up.ndim == 1 and filter_up.shape[0] == 1:
                filter_up = filter_up.square()[None]
            if down == 1 and filter_down.ndim == 1 and filter_down.shape[
                    0] == 1:
                filter_down = filter_down.square()[None]

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
                    input, filter_up, filter_down, bias, si.to(input.device),
                    up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp,
                    flip_filter, write_signs)
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
                    filter=filter_up,
                    up=up,
                    padding=[px0, px1, py0, py1],
                    gain=float(up**2),
                    flip_filter=flip_filter)  # Upsample.
                # Activation function and sign handling. Modifies y in-place.
                so = ext_module.filtered_lrelu_act_(y, si.to(y.device), sx, sy,
                                                    gain, slope, clamp,
                                                    write_signs)
                y = upfirdn2d(
                    input=y,
                    filter=filter_down,
                    down=down,
                    flip_filter=flip_filter)  # Downsample.

            # Prepare for gradient computation.
            ctx.save_for_backward(filter_up, filter_down,
                                  (si if si.numel() else so))
            ctx.x_shape = input.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            filter_up, filter_down, si = ctx.saved_tensors
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
                    (filter_up.shape[-1] - 1) + (filter_down.shape[-1] - 1) -
                    px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (filter_up.shape[0] - 1) + (filter_down.shape[0] - 1) -
                    py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up**2) / (down**2)
                ff = (not flip_filter)
                sx = sx - (filter_up.shape[-1] - 1) + px0
                sy = sy - (filter_up.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(
                    up=down,
                    down=up,
                    padding=pp,
                    gain=gg,
                    slope=slope,
                    clamp=None,
                    flip_filter=ff).apply(dy, filter_down, filter_up, None, si,
                                          sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda




_filtered_lrelu_musa_cache: Dict = dict()


def _filtered_lrelu_musa(up: int = 1,
                         down: int = 1,
                         padding: int = 0,
                         gain: float = np.sqrt(2),
                         slope: float = 0.2,
                         clamp: Optional[Union[float, int]] = None,
                         flip_filter: bool = False):
    """Fast MUSA implementation of `filtered_lrelu()` using custom ops.

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
        clamp (float or int): Maximum magnitude for leaky ReLU
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
    if key in _filtered_lrelu_musa_cache:
        return _filtered_lrelu_musa_cache[key]

    # Forward op.
    class FilteredLReluMusa(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, filter_up, filter_down, bias, si, sx, sy):
            # pylint: disable=arguments-differ
            assert isinstance(input, torch.Tensor) and input.ndim == 4

            # Replace empty up/downsample kernels with full 1x1 kernels
            # (faster than separable).
            if filter_up is None:
                filter_up = torch.ones([1, 1],
                                       dtype=torch.float32,
                                       device=input.device)
            if filter_down is None:
                filter_down = torch.ones([1, 1],
                                         dtype=torch.float32,
                                         device=input.device)
            assert 1 <= filter_up.ndim <= 2
            assert 1 <= filter_down.ndim <= 2

            # Replace separable 1x1 kernels with full 1x1 kernels when scale
            # factor is 1.
            if up == 1 and filter_up.ndim == 1 and filter_up.shape[0] == 1:
                filter_up = filter_up.square()[None]
            if down == 1 and filter_down.ndim == 1 and filter_down.shape[
                    0] == 1:
                filter_down = filter_down.square()[None]

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

            # Call C++/MUSA plugin if datatype is supported.
            if input.dtype in [torch.float16, torch.float32]:
                if torch.musa.current_stream(
                        input.device) != torch.musa.default_stream(
                            input.device):
                    warnings.warn(
                        'filtered_lrelu called with non-default musa stream '
                        'but concurrent execution is not supported',
                        RuntimeWarning)
                y, so, return_code = ext_module.filtered_lrelu(
                    input, filter_up, filter_down, bias, si.to(input.device),
                    up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp,
                    flip_filter, write_signs)
            else:
                return_code = -1

            # No musa kernel found? Fall back to generic implementation.
            # Still more memory efficient than the reference implementation
            # because only the bit-packed sign tensor is retained for gradient
            # computation.
            if return_code < 0:
                warnings.warn(
                    'filtered_lrelu called with parameters that have no '
                    'optimized musa kernel, using generic fallback',
                    RuntimeWarning)

                y = input.add(bias.unsqueeze(-1).unsqueeze(-1))  # Add bias.
                y = upfirdn2d(
                    input=y,
                    filter=filter_up,
                    up=up,
                    padding=[px0, px1, py0, py1],
                    gain=float(up**2),
                    flip_filter=flip_filter)  # Upsample.
                # Activation function and sign handling. Modifies y in-place.
                so = ext_module.filtered_lrelu_act_(y, si.to(y.device), sx, sy,
                                                    gain, slope, clamp,
                                                    write_signs)
                y = upfirdn2d(
                    input=y,
                    filter=filter_down,
                    down=down,
                    flip_filter=flip_filter)  # Downsample.

            # Prepare for gradient computation.
            ctx.save_for_backward(filter_up, filter_down,
                                  (si if si.numel() else so))
            ctx.x_shape = input.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = sx, sy
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            filter_up, filter_down, si = ctx.saved_tensors
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
                    (filter_up.shape[-1] - 1) + (filter_down.shape[-1] - 1) -
                    px0,
                    xw * up - yw * down + px0 - (up - 1),
                    (filter_up.shape[0] - 1) + (filter_down.shape[0] - 1) -
                    py0,
                    xh * up - yh * down + py0 - (up - 1),
                ]
                gg = gain * (up**2) / (down**2)
                ff = (not flip_filter)
                sx = sx - (filter_up.shape[-1] - 1) + px0
                sy = sy - (filter_up.shape[0] - 1) + py0
                dx = _filtered_lrelu_musa(
                    up=down,
                    down=up,
                    padding=pp,
                    gain=gg,
                    slope=slope,
                    clamp=None,
                    flip_filter=ff).apply(dy, filter_down, filter_up, None, si,
                                          sx, sy)

            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])

            return dx, dfu, dfd, db, dsi, dsx, dsy

    # Add to cache.
    _filtered_lrelu_musa_cache[key] = FilteredLReluMusa
    return FilteredLReluMusa
