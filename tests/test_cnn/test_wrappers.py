# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mmcv.cnn.bricks import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                             Linear, MaxPool2d, MaxPool3d)

if torch.__version__ != 'parrots':
    torch_version = '1.1'
else:
    torch_version = 'parrots'


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_channel,out_channel,kernel_size,stride,padding,dilation',
    [(10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 3, 3, 5, 2, 1, 2)])
def test_conv2d(in_w, in_h, in_channel, out_channel, kernel_size, stride,
                padding, dilation):
    """
    CommandLine:
        xdoctest -m tests/test_wrappers.py test_conv2d
    """
    # train mode
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_h, in_w)
    torch.manual_seed(0)
    wrapper = Conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_h, in_w).requires_grad_(True)
    torch.manual_seed(0)
    ref = nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    wrapper_out.sum().backward()
    assert wrapper.weight.grad is not None
    assert wrapper.weight.grad.shape == wrapper.weight.shape

    assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_channel, in_h, in_w)
    wrapper = Conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    wrapper.eval()
    wrapper(x_empty)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_t,in_channel,out_channel,kernel_size,stride,padding,dilation',  # noqa: E501
    [(10, 10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 20, 3, 3, 5, 2, 1, 2)])
def test_conv3d(in_w, in_h, in_t, in_channel, out_channel, kernel_size, stride,
                padding, dilation):
    """
    CommandLine:
        xdoctest -m tests/test_wrappers.py test_conv3d
    """
    # train mode
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_t, in_h, in_w)
    torch.manual_seed(0)
    wrapper = Conv3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_t, in_h,
                           in_w).requires_grad_(True)
    torch.manual_seed(0)
    ref = nn.Conv3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    wrapper_out.sum().backward()
    assert wrapper.weight.grad is not None
    assert wrapper.weight.grad.shape == wrapper.weight.shape

    assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_channel, in_t, in_h, in_w)
    wrapper = Conv3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation)
    wrapper.eval()
    wrapper(x_empty)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_channel,out_channel,kernel_size,stride,padding,dilation',
    [(10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 3, 3, 5, 2, 1, 2)])
def test_conv_transposed_2d(in_w, in_h, in_channel, out_channel, kernel_size,
                            stride, padding, dilation):
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_h, in_w, requires_grad=True)
    # out padding must be smaller than either stride or dilation
    op = min(stride, dilation) - 1
    if torch.__version__ == 'parrots':
        op = 0
    torch.manual_seed(0)
    wrapper = ConvTranspose2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_h, in_w)
    torch.manual_seed(0)
    ref = nn.ConvTranspose2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    wrapper_out.sum().backward()
    assert wrapper.weight.grad is not None
    assert wrapper.weight.grad.shape == wrapper.weight.shape

    assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_channel, in_h, in_w)
    wrapper = ConvTranspose2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    wrapper.eval()
    wrapper(x_empty)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_t,in_channel,out_channel,kernel_size,stride,padding,dilation',  # noqa: E501
    [(10, 10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 20, 3, 3, 5, 2, 1, 2)])
def test_conv_transposed_3d(in_w, in_h, in_t, in_channel, out_channel,
                            kernel_size, stride, padding, dilation):
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_t, in_h, in_w, requires_grad=True)
    # out padding must be smaller than either stride or dilation
    op = min(stride, dilation) - 1
    torch.manual_seed(0)
    wrapper = ConvTranspose3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_t, in_h, in_w)
    torch.manual_seed(0)
    ref = nn.ConvTranspose3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    wrapper_out.sum().backward()
    assert wrapper.weight.grad is not None
    assert wrapper.weight.grad.shape == wrapper.weight.shape

    assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_channel, in_t, in_h, in_w)
    wrapper = ConvTranspose3d(
        in_channel,
        out_channel,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        output_padding=op)
    wrapper.eval()
    wrapper(x_empty)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_channel,out_channel,kernel_size,stride,padding,dilation',
    [(10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 3, 3, 5, 2, 1, 2)])
def test_max_pool_2d(in_w, in_h, in_channel, out_channel, kernel_size, stride,
                     padding, dilation):
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_h, in_w, requires_grad=True)
    wrapper = MaxPool2d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_h, in_w)
    ref = nn.MaxPool2d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    assert torch.equal(wrapper(x_normal), ref_out)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize(
    'in_w,in_h,in_t,in_channel,out_channel,kernel_size,stride,padding,dilation',  # noqa: E501
    [(10, 10, 10, 1, 1, 3, 1, 0, 1), (20, 20, 20, 3, 3, 5, 2, 1, 2)])
@pytest.mark.skipif(
    torch.__version__ == 'parrots' and not torch.cuda.is_available(),
    reason='parrots requires CUDA support')
def test_max_pool_3d(in_w, in_h, in_t, in_channel, out_channel, kernel_size,
                     stride, padding, dilation):
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_channel, in_t, in_h, in_w, requires_grad=True)
    wrapper = MaxPool3d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)
    if torch.__version__ == 'parrots':
        x_empty = x_empty.cuda()
    wrapper_out = wrapper(x_empty)
    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_channel, in_t, in_h, in_w)
    ref = nn.MaxPool3d(
        kernel_size, stride=stride, padding=padding, dilation=dilation)
    if torch.__version__ == 'parrots':
        x_normal = x_normal.cuda()
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    assert torch.equal(wrapper(x_normal), ref_out)


@patch('torch.__version__', torch_version)
@pytest.mark.parametrize('in_w,in_h,in_feature,out_feature', [(10, 10, 1, 1),
                                                              (20, 20, 3, 3)])
def test_linear(in_w, in_h, in_feature, out_feature):
    # wrapper op with 0-dim input
    x_empty = torch.randn(0, in_feature, requires_grad=True)
    torch.manual_seed(0)
    wrapper = Linear(in_feature, out_feature)
    wrapper_out = wrapper(x_empty)

    # torch op with 3-dim input as shape reference
    x_normal = torch.randn(3, in_feature)
    torch.manual_seed(0)
    ref = nn.Linear(in_feature, out_feature)
    ref_out = ref(x_normal)

    assert wrapper_out.shape[0] == 0
    assert wrapper_out.shape[1:] == ref_out.shape[1:]

    wrapper_out.sum().backward()
    assert wrapper.weight.grad is not None
    assert wrapper.weight.grad.shape == wrapper.weight.shape

    assert torch.equal(wrapper(x_normal), ref_out)

    # eval mode
    x_empty = torch.randn(0, in_feature)
    wrapper = Linear(in_feature, out_feature)
    wrapper.eval()
    wrapper(x_empty)


@patch('mmcv.cnn.bricks.wrappers.TORCH_VERSION', (1, 10))
def test_nn_op_forward_called():
    for m in ['Conv2d', 'ConvTranspose2d', 'MaxPool2d']:
        with patch(f'torch.nn.{m}.forward') as nn_module_forward:
            # randn input
            x_empty = torch.randn(0, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_empty)
            nn_module_forward.assert_called_with(x_empty)

            # non-randn input
            x_normal = torch.randn(1, 3, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_normal)
            nn_module_forward.assert_called_with(x_normal)

    for m in ['Conv3d', 'ConvTranspose3d', 'MaxPool3d']:
        with patch(f'torch.nn.{m}.forward') as nn_module_forward:
            # randn input
            x_empty = torch.randn(0, 3, 10, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_empty)
            nn_module_forward.assert_called_with(x_empty)

            # non-randn input
            x_normal = torch.randn(1, 3, 10, 10, 10)
            wrapper = eval(m)(3, 2, 1)
            wrapper(x_normal)
            nn_module_forward.assert_called_with(x_normal)

    with patch('torch.nn.Linear.forward') as nn_module_forward:
        # randn input
        x_empty = torch.randn(0, 3)
        wrapper = Linear(3, 3)
        wrapper(x_empty)
        nn_module_forward.assert_called_with(x_empty)

        # non-randn input
        x_normal = torch.randn(1, 3)
        wrapper = Linear(3, 3)
        wrapper(x_normal)
        nn_module_forward.assert_called_with(x_normal)
