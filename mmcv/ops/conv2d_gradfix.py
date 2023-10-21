# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# source: https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/conv2d_gradfix.py # noqa
"""Custom replacement for `torch.nn.functional.conv2d` that supports
arbitrarily high order gradients with zero performance penalty."""

import contextlib
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
from mmengine.utils import digit_version
from mmengine.utils.dl_utils.parrots_wrapper import is_rocm_pytorch

enabled = True
weight_gradients_disabled = False


@contextlib.contextmanager
def no_weight_gradients(disable=True):
    global weight_gradients_disabled
    old = weight_gradients_disabled
    if disable:
        weight_gradients_disabled = True
    yield
    weight_gradients_disabled = old


def conv2d(input: torch.Tensor,
           weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None,
           stride: Union[int, Tuple[int, ...]] = 1,
           padding: Union[int, Tuple[int, ...]] = 0,
           dilation: Union[int, Tuple[int, ...]] = 1,
           groups: int = 1):
    flag = True
    if digit_version(torch.__version__) >= digit_version('1.10.0'):
        warnings.warn('Since '
                      'aten:cudnn_convolution_backward_weight is '
                      f'not supported in torch=={torch.__version__},'
                      ' rolling back to `torch.nn.functional.conv2d`')
        flag = False
    if _should_use_custom_op(input) and flag:
        return _conv2d_gradfix(
            transpose=False,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups).apply(input, weight, bias)
    return torch.nn.functional.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups)


def conv_transpose2d(input: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None,
                     stride: Union[int, Tuple[int, ...]] = 1,
                     padding: Union[int, Tuple[int, ...]] = 0,
                     output_padding: Union[int, Tuple[int, ...]] = 0,
                     groups: int = 1,
                     dilation: Union[int, Tuple[int, ...]] = 1):
    if _should_use_custom_op(input):
        return _conv2d_gradfix(
            transpose=True,
            weight_shape=weight.shape,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation).apply(input, weight, bias)
    return torch.nn.functional.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation)


def _should_use_custom_op(input):
    assert isinstance(input, torch.Tensor)
    if (not enabled) or (not torch.backends.cudnn.enabled):
        return False
    if input.device.type != 'cuda':
        return False
    return True


def _to_tuple(x, ndim):
    xs = tuple(x) if isinstance(x, (tuple, list)) else (x, ) * ndim
    assert len(xs) == ndim
    assert all(isinstance(x, int) for x in xs)
    return xs


_conv2d_gradfix_cache: Dict = dict()
_null_tensor = torch.empty([0])


def _conv2d_gradfix(
    transpose: bool,
    weight_shape: Tuple[int, ...],
    stride: Union[int, Tuple[int, ...]],
    padding: Union[int, Tuple[int, ...]],
    output_padding: Union[int, Tuple[int, ...]],
    dilation: Union[int, Tuple[int, ...]],
    groups: int,
):
    # Parse arguments.
    ndim = 2
    weight_shape = tuple(weight_shape)
    stride = _to_tuple(stride, ndim)
    padding = _to_tuple(padding, ndim)
    output_padding = _to_tuple(output_padding, ndim)
    dilation = _to_tuple(dilation, ndim)

    # Lookup from cache.
    key = (transpose, weight_shape, stride, padding, output_padding, dilation,
           groups)
    if key in _conv2d_gradfix_cache:
        return _conv2d_gradfix_cache[key]

    # Validate arguments.

    assert groups >= 1
    assert len(weight_shape) == ndim + 2
    assert all(stride[i] >= 1 for i in range(ndim))  # type: ignore
    assert all(padding[i] >= 0 for i in range(ndim))  # type: ignore
    assert all(dilation[i] >= 0 for i in range(ndim))  # type: ignore
    if not transpose:
        assert all(output_padding[i] == 0 for i in range(ndim))  # type: ignore
    else:  # transpose
        for i in range(ndim):
            assert 0 <= output_padding[i] < max(  # type: ignore
                stride[i],  # type: ignore
                dilation[i])  # type: ignore

    # Helpers.
    common_kwargs = dict(
        stride=stride, padding=padding, dilation=dilation, groups=groups)

    def calc_output_padding(input_shape, output_shape):
        if transpose:
            return [0, 0]
        return [
            input_shape[i + 2] - (output_shape[i + 2] - 1) * stride[i] -
            (1 - 2 * padding[i]) - dilation[i] * (weight_shape[i + 2] - 1)
            for i in range(ndim)
        ]

    # Forward & backward.
    class Conv2d(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, weight, bias):
            assert weight.shape == weight_shape
            ctx.save_for_backward(
                input if weight.requires_grad else _null_tensor,
                weight if input.requires_grad else _null_tensor,
            )
            ctx.input_shape = input.shape

            # Simple 1x1 convolution => cuBLAS (only on Volta, not on Ampere).
            if weight_shape[2:] == stride == dilation == (
                    1, 1) and padding == (
                        0, 0) and torch.cuda.get_device_capability(
                            input.device) < (8, 0):
                a = weight.reshape(groups, weight_shape[0] // groups,
                                   weight_shape[1])
                b = input.reshape(input.shape[0], groups,
                                  input.shape[1] // groups, -1)
                c = (a.transpose(1, 2) if transpose else a) @ b.permute(
                    1, 2, 0, 3).flatten(2)
                c = c.reshape(-1, input.shape[0],
                              *input.shape[2:]).transpose(0, 1)
                c = c if bias is None else c + bias.unsqueeze(0).unsqueeze(
                    2).unsqueeze(3)
                return c.contiguous(
                    memory_format=(torch.channels_last if input.stride(1) ==
                                   1 else torch.contiguous_format))

            # General case => cuDNN.
            if transpose:
                return torch.nn.functional.conv_transpose2d(
                    input=input,
                    weight=weight,
                    bias=bias,
                    output_padding=output_padding,
                    **common_kwargs)
            return torch.nn.functional.conv2d(
                input=input, weight=weight, bias=bias, **common_kwargs)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight = ctx.saved_tensors
            input_shape = ctx.input_shape
            grad_input = None
            grad_weight = None
            grad_bias = None

            if ctx.needs_input_grad[0]:
                p = calc_output_padding(
                    input_shape=input_shape, output_shape=grad_output.shape)
                op = _conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs)
                grad_input = op.apply(grad_output, weight, None)
                assert grad_input.shape == input_shape

            if ctx.needs_input_grad[1] and not weight_gradients_disabled:
                grad_weight = Conv2dGradWeight.apply(grad_output, input)
                assert grad_weight.shape == weight_shape

            if ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum([0, 2, 3])

            return grad_input, grad_weight, grad_bias

    # Gradient with respect to the weights.
    class Conv2dGradWeight(torch.autograd.Function):

        @staticmethod
        def forward(ctx, grad_output, input):
            ctx.save_for_backward(
                grad_output if input.requires_grad else _null_tensor,
                input if grad_output.requires_grad else _null_tensor,
            )
            ctx.grad_output_shape = grad_output.shape
            ctx.input_shape = input.shape

            # Simple 1x1 convolution => cuBLAS (on both Volta and Ampere).
            if weight_shape[2:] == stride == dilation == (
                    1, 1) and padding == (0, 0):
                a = grad_output.reshape(grad_output.shape[0], groups,
                                        grad_output.shape[1] // groups,
                                        -1).permute(1, 2, 0, 3).flatten(2)
                b = input.reshape(input.shape[0], groups,
                                  input.shape[1] // groups,
                                  -1).permute(1, 2, 0, 3).flatten(2)
                c = (b @ a.transpose(1, 2) if transpose else
                     a @ b.transpose(1, 2)).reshape(weight_shape)
                return c.contiguous(
                    memory_format=(torch.channels_last if input.stride(1) ==
                                   1 else torch.contiguous_format))

            # PyTorch consolidated convolution backward API in PR:
            # https://github.com/pytorch/pytorch/commit/3dc3651e0ee3623f669c3a2c096408dbc476d122  # noqa: E501
            # Enhance the code referring to the discussion:
            # https://github.com/pytorch/pytorch/issues/74437
            if digit_version(torch.__version__) >= digit_version('1.11.0'):
                empty_weight = torch.tensor(
                    0.0, dtype=input.dtype,
                    device=input.device).expand(weight_shape)
                output_padding = calc_output_padding(input.shape,
                                                     grad_output.shape)
                return torch.ops.aten.convolution_backward(
                    grad_output,
                    input,
                    empty_weight,
                    None,
                    stride=stride,
                    dilation=dilation,
                    transposed=transpose,
                    padding=padding,
                    groups=groups,
                    output_padding=output_padding,
                    output_mask=[0, 1, 0])[1]
            else:
                if is_rocm_pytorch():
                    name = 'aten::miopen_convolution_transpose_backward_weight'
                    if not transpose:
                        name = 'aten::miopen_convolution_backward_weight'
                    flags = [
                        torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic
                    ]
                else:
                    # General case => cuDNN.
                    name = ('aten::cudnn_convolution_transpose_backward_weight'
                            if transpose else
                            'aten::cudnn_convolution_backward_weight')
                    flags = [
                        torch.backends.cudnn.benchmark,
                        torch.backends.cudnn.deterministic,
                        torch.backends.cudnn.allow_tf32
                    ]
                return torch._C._jit_get_operation(name)(weight_shape,
                                                         grad_output, input,
                                                         padding, stride,
                                                         dilation, groups,
                                                         *flags)

        @staticmethod
        def backward(ctx, grad2_grad_weight):
            grad_output, input = ctx.saved_tensors
            grad_output_shape = ctx.grad_output_shape
            input_shape = ctx.input_shape
            grad2_grad_output = None
            grad2_input = None

            if ctx.needs_input_grad[0]:
                grad2_grad_output = Conv2d.apply(input, grad2_grad_weight,
                                                 None)
                assert grad2_grad_output.shape == grad_output_shape

            if ctx.needs_input_grad[1]:
                p = calc_output_padding(
                    input_shape=input_shape, output_shape=grad_output_shape)
                op = _conv2d_gradfix(
                    transpose=(not transpose),
                    weight_shape=weight_shape,
                    output_padding=p,
                    **common_kwargs)
                grad2_input = op.apply(grad_output, grad2_grad_weight, None)
                assert grad2_input.shape == input_shape

            return grad2_grad_output, grad2_input

    _conv2d_gradfix_cache[key] = Conv2d
    return Conv2d
