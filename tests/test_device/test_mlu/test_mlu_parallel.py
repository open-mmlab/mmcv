# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mmcv.device.mlu import MLUDataParallel, MLUDistributedDataParallel
from mmcv.device.mlu._functions import Scatter, get_input_device, scatter
from mmcv.parallel import is_module_wrapper


def mock(*args, **kwargs):
    pass


mlu_is_available = False
if hasattr(torch, 'mlu'):
    import torch_mlu  # noqa: F401
    mlu_is_available = torch.mlu.is_available()


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', mock)
def test_is_module_wrapper():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

    model = Model()
    assert not is_module_wrapper(model)

    if mlu_is_available:
        mludp = MLUDataParallel(model)
        assert is_module_wrapper(mludp)

        mluddp = MLUDistributedDataParallel(model, process_group=MagicMock())
        assert is_module_wrapper(mluddp)


def test_get_input_device():
    # if the device is CPU, return -1
    input = torch.zeros([1, 3, 3, 3])
    assert get_input_device(input) == -1
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    assert get_input_device(inputs) == -1

    # if the device is MLU, return the index of device
    if mlu_is_available:
        input = torch.zeros([1, 3, 3, 3]).to('mlu')
        assert get_input_device(input) == 'mlu'
        inputs = [
            torch.zeros([1, 3, 3, 3]).to('mlu'),
            torch.zeros([1, 4, 4, 4]).to('mlu')
        ]
        assert get_input_device(inputs) == 'mlu'

    # input should be a tensor or list of tensor
    with pytest.raises(Exception):
        get_input_device(5)


def test_scatter():
    # if the device is CPU, just return the input
    input = torch.zeros([1, 3, 3, 3])
    output = scatter(input=input, devices=[-1])
    assert torch.allclose(input, output)

    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = scatter(input=inputs, devices=[-1])
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is MLU, copy the input from CPU to MLU
    if mlu_is_available:
        input = torch.zeros([1, 3, 3, 3])
        output = scatter(input=input, devices=[0])
        assert torch.allclose(input.to('mlu'), output)

        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = scatter(input=inputs, devices=[0])
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mlu'), output)

    # input should be a tensor or list of tensor
    with pytest.raises(Exception):
        scatter(5, [-1])


def test_Scatter():
    # if the device is CPU, just return the input
    target_mlus = [-1]
    input = torch.zeros([1, 3, 3, 3])
    outputs = Scatter.forward(target_mlus, input)
    assert isinstance(outputs, tuple)
    assert torch.allclose(input, outputs[0])

    target_mlus = [-1]
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = Scatter.forward(target_mlus, inputs)
    assert isinstance(outputs, tuple)
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is MLU, copy the input from CPU to MLU
    if mlu_is_available:
        target_mlus = [0]
        input = torch.zeros([1, 3, 3, 3])
        outputs = Scatter.forward(target_mlus, input)
        assert isinstance(outputs, tuple)
        assert torch.allclose(input.to('mlu'), outputs[0])

        target_mlus = [0]
        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = Scatter.forward(target_mlus, inputs)
        assert isinstance(outputs, tuple)
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mlu'), output[0])
