# Modified from
# https://github.com/NVlabs/stylegan3/blob/main/torch_utils/ops/bias_act.py

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# source: https://github.com/open-mmlab/mmediting/blob/dev-1.x/mmedit/models/editors/stylegan3/stylegan3_ops/ops/bias_act.py # noqa
"""Custom PyTorch ops for efficient bias and activation."""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['bias_act'])


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the
    attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


activation_funcs = {
    'linear':
    EasyDict(
        func=lambda x, **_: x,
        def_alpha=0,
        def_gain=1,
        cuda_idx=1,
        ref='',
        has_2nd_grad=False),
    'relu':
    EasyDict(
        func=lambda x, **_: torch.nn.functional.relu(x),
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=2,
        ref='y',
        has_2nd_grad=False),
    'lrelu':
    EasyDict(
        func=lambda x, alpha, **_: torch.nn.functional.leaky_relu(x, alpha),
        def_alpha=0.2,
        def_gain=np.sqrt(2),
        cuda_idx=3,
        ref='y',
        has_2nd_grad=False),
    'tanh':
    EasyDict(
        func=lambda x, **_: torch.tanh(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=4,
        ref='y',
        has_2nd_grad=True),
    'sigmoid':
    EasyDict(
        func=lambda x, **_: torch.sigmoid(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=5,
        ref='y',
        has_2nd_grad=True),
    'elu':
    EasyDict(
        func=lambda x, **_: torch.nn.functional.elu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=6,
        ref='y',
        has_2nd_grad=True),
    'selu':
    EasyDict(
        func=lambda x, **_: torch.nn.functional.selu(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=7,
        ref='y',
        has_2nd_grad=True),
    'softplus':
    EasyDict(
        func=lambda x, **_: torch.nn.functional.softplus(x),
        def_alpha=0,
        def_gain=1,
        cuda_idx=8,
        ref='y',
        has_2nd_grad=True),
    'swish':
    EasyDict(
        func=lambda x, **_: torch.sigmoid(x) * x,
        def_alpha=0,
        def_gain=np.sqrt(2),
        cuda_idx=9,
        ref='x',
        has_2nd_grad=True),
}

_plugin = None
_null_tensor = torch.empty([0])


def bias_act(x: torch.Tensor,
             b: Optional[torch.Tensor] = None,
             dim: int = 1,
             act: str = 'linear',
             alpha: Optional[Union[float, int]] = None,
             gain: Optional[float] = None,
             clamp: Optional[float] = None,
             impl: str = 'cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function
    `act`, and scales the result by `gain`. Each of the steps is optional.
    In most cases, the fused op is considerably more efficient than performing
    the same calculation using standard PyTorch ops. It supports first and
    second order gradients, but not third order gradients.

    Args:
        x (torch.Tensor): Input activation tensor. Can be of any shape.
        b (Optional[torch.Tensor]): Bias vector, or `None` to disable.
            Must be a 1D tensor of the same type as `x`. The shape must be
            known, and it must match the dimension of `x` corresponding to
            `dim`. Defaults to None.
        dim (int): The dimension in `x` corresponding to the elements of `b`.
            The value of `dim` is ignored if `b` is not specified.
            Defaults to 1.
        act (str): Name of the activation function to evaluate, or `"linear"`
            to disable. Can be e.g. "relu", "lrelu", "tanh", "sigmoid",
            "swish", etc. See `activation_funcs` for a full list. `None` is not
            allowed. Defaults to `linear`.
        alpha (Optional[Union[float, int]]): Shape parameter for the activation
            function, or `None` to use the default. Defaults to None.
        gain (Optional[float]): Scaling factor for the output tensor, or `None`
            to use default. See `activation_funcs` for the default scaling of
            each activation function. If unsure, consider specifying 1.
            Defaults to None.
        clamp (Optional[float]):  Clamp the output values to `[-clamp, +clamp]`
            , or `None` to disable the clamping (default). Defaults to None.
        impl (str): Name of the implementation to use. Can be `"ref"` or
            `"cuda"`. Defaults to "cuda".

    Returns:
        torch.Tensor: Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.is_cuda:
        return _bias_act_cuda(
            dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp).apply(x, b)
    return _bias_act_ref(
        x=x, b=b, dim=dim, act=act, alpha=alpha, gain=gain, clamp=clamp)


def _bias_act_ref(x: torch.Tensor,
                  b: Optional[torch.Tensor] = None,
                  dim: int = 1,
                  act: str = 'linear',
                  alpha: Optional[Union[float, int]] = None,
                  gain: Optional[float] = None,
                  clamp: Optional[float] = None):
    """Slow reference implementation of `bias_act()` using standard PyTorch
    ops.

    Adds bias `b` to activation tensor `x`, evaluates activation function
    `act`, and scales the result by `gain`. Each of the steps is optional.
    In most cases, the fused op is considerably more efficient than performing
    the same calculation using standard PyTorch ops. It supports first and
    second order gradients, but not third order gradients.

    Args:
        x (torch.Tensor): Input activation tensor. Can be of any shape.
        b (Optional[torch.Tensor]): Bias vector, or `None` to disable.
            Must be a 1D tensor of the same type as `x`. The shape must be
            known, and it must match the dimension of `x` corresponding to
            `dim`. Defaults to None.
        dim (int): The dimension in `x` corresponding to the elements of `b`.
            The value of `dim` is ignored if `b` is not specified.
            Defaults to 1.
        act (str): Name of the activation function to evaluate, or `"linear"`
            to disable. Can be e.g. "relu", "lrelu", "tanh", "sigmoid",
            "swish", etc. See `activation_funcs` for a full list. `None` is not
            allowed. Defaults to `linear`.
        alpha (Optional[Union[float, int]]): Shape parameter for the activation
            function, or `None` to use the default. Defaults to None.
        gain (Optional[float]): Scaling factor for the output tensor, or `None`
            to use default. See `activation_funcs` for the default scaling of
            each activation function. If unsure, consider specifying 1.
            Defaults to None.
        clamp (Optional[float]):  Clamp the output values to
            `[-clamp, +clamp]`, or `None` to disable the clamping (default).
            Defaults to None.

    Returns:
        torch.Tensor: Tensor of the same shape and datatype as `x`.
    """
    assert isinstance(x, torch.Tensor)
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias.
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = float(alpha)
    x = spec.func(x, alpha=alpha)

    # Scale by gain.
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # Clamp.
    if clamp >= 0:
        # pylint: disable=invalid-unary-operand-type
        x = x.clamp(-clamp, clamp)
    return x


_bias_act_cuda_cache: Dict = dict()


def _bias_act_cuda(dim: int = 1,
                   act: str = 'linear',
                   alpha: Optional[Union[float, int]] = None,
                   gain: Optional[float] = None,
                   clamp: Optional[float] = None):
    """"Fast CUDA implementation of `bias_act()` using custom ops.

    Args:
        dim (int): The dimension in `x` corresponding to the elements of `b`.
            The value of `dim` is ignored if `b` is not specified.
            Defaults to 1.
        act (str): Name of the activation function to evaluate, or `"linear"`
            to disable. Can be e.g. "relu", "lrelu", "tanh", "sigmoid",
            "swish", etc. See `activation_funcs` for a full list. `None` is not
            allowed. Defaults to `linear`.
        alpha (Optional[Union[float, int]]): Shape parameter for the activation
            function, or `None` to use the default. Defaults to None.
        gain (Optional[float]): Scaling factor for the output tensor, or `None`
            to use default. See `activation_funcs` for the default scaling of
            each activation function. If unsure, consider specifying 1.
            Defaults to None.
        clamp (Optional[float]): Clamp the output values to `[-clamp, +clamp]`,
            or `None` to disable the clamping (default). Defaults to None.

    Returns:
        torch.Tensor: Tensor of the same shape and datatype as `x`.
    """
    # Parse arguments.
    assert clamp is None or clamp >= 0
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Lookup from cache.
    key = (dim, act, alpha, gain, clamp)
    if key in _bias_act_cuda_cache:
        return _bias_act_cuda_cache[key]

    # Forward op.
    class BiasActCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, b):  # pylint: disable=arguments-differ
            ctx.memory_format = torch.channels_last if x.ndim > 2 and x.stride(
                1) == 1 else torch.contiguous_format
            x = x.contiguous(memory_format=ctx.memory_format)
            b = b.contiguous() if b is not None else _null_tensor.to(x.device)
            y = x
            if act != 'linear' or gain != 1 or clamp >= 0 or (
                    b is not _null_tensor.to(x.device)):
                y = ext_module.bias_act(x, b, _null_tensor.to(x.device),
                                        _null_tensor.to(x.device),
                                        _null_tensor.to(x.device), 0, dim,
                                        spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(
                x if 'x' in spec.ref or spec.has_2nd_grad else _null_tensor.to(
                    x.device), b if 'x' in spec.ref or spec.has_2nd_grad else
                _null_tensor.to(x.device),
                y if 'y' in spec.ref else _null_tensor.to(x.device))
            return y

        @staticmethod
        def backward(ctx, dy):  # pylint: disable=arguments-differ
            dy = dy.contiguous(memory_format=ctx.memory_format)
            x, b, y = ctx.saved_tensors
            dx = None
            db = None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                dx = dy
                if act != 'linear' or gain != 1 or clamp >= 0:
                    dx = BiasActCudaGrad.apply(dy, x, b, y)

            if ctx.needs_input_grad[1]:
                db = dx.sum([i for i in range(dx.ndim) if i != dim])

            return dx, db

    # Backward op.
    class BiasActCudaGrad(torch.autograd.Function):

        @staticmethod
        def forward(ctx, dy, x, b, y):  # pylint: disable=arguments-differ
            ctx.memory_format = torch.channels_last if dy.ndim > 2 and (
                dy.stride(1) == 1) else torch.contiguous_format
            dx = ext_module.bias_act(dy, b, x, y, _null_tensor.to(x.device), 1,
                                     dim, spec.cuda_idx, alpha, gain, clamp)
            ctx.save_for_backward(
                dy if spec.has_2nd_grad else _null_tensor.to(x.device), x, b,
                y)
            return dx

        @staticmethod
        def backward(ctx, d_dx):  # pylint: disable=arguments-differ
            d_dx = d_dx.contiguous(memory_format=ctx.memory_format)
            dy, x, b, y = ctx.saved_tensors
            d_dy = None
            d_x = None
            d_b = None
            d_y = None

            if ctx.needs_input_grad[0]:
                d_dy = BiasActCudaGrad.apply(d_dx, x, b, y)

            if spec.has_2nd_grad and (ctx.needs_input_grad[1]
                                      or ctx.needs_input_grad[2]):
                d_x = ext_module.bias_act(d_dx, b, x, y, dy, 2, dim,
                                          spec.cuda_idx, alpha, gain, clamp)

            if spec.has_2nd_grad and ctx.needs_input_grad[2]:
                d_b = d_x.sum([i for i in range(d_x.ndim) if i != dim])

            return d_dy, d_x, d_b, d_y

    # Add to cache.
    _bias_act_cuda_cache[key] = BiasActCuda
    return BiasActCuda