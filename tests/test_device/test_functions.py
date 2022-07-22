# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.device._functions import Scatter, scatter
from mmcv.utils import IS_MLU_AVAILABLE, IS_MPS_AVAILABLE


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
    if IS_MLU_AVAILABLE:
        input = torch.zeros([1, 3, 3, 3])
        output = scatter(input=input, devices=[0])
        assert torch.allclose(input.to('mlu'), output)

        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = scatter(input=inputs, devices=[0])
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mlu'), output)

    # if the device is MPS, copy the input from CPU to MPS
    if IS_MPS_AVAILABLE:
        input = torch.zeros([1, 3, 3, 3])
        output = scatter(input=input, devices=[0])
        assert torch.allclose(input.to('mps'), output)

        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = scatter(input=inputs, devices=[0])
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mps'), output)

    # input should be a tensor or list of tensor
    with pytest.raises(Exception):
        scatter(5, [-1])


def test_Scatter():
    # if the device is CPU, just return the input
    target_devices = [-1]
    input = torch.zeros([1, 3, 3, 3])
    outputs = Scatter.forward(target_devices, input)
    assert isinstance(outputs, tuple)
    assert torch.allclose(input, outputs[0])

    target_devices = [-1]
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = Scatter.forward(target_devices, inputs)
    assert isinstance(outputs, tuple)
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is MLU, copy the input from CPU to MLU
    if IS_MLU_AVAILABLE:
        target_devices = [0]
        input = torch.zeros([1, 3, 3, 3])
        outputs = Scatter.forward(target_devices, input)
        assert isinstance(outputs, tuple)
        assert torch.allclose(input.to('mlu'), outputs[0])

        target_devices = [0]
        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = Scatter.forward(target_devices, inputs)
        assert isinstance(outputs, tuple)
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mlu'), output[0])

    # if the device is MPS, copy the input from CPU to MPS
    if IS_MPS_AVAILABLE:
        target_devices = [0]
        input = torch.zeros([1, 3, 3, 3])
        outputs = Scatter.forward(target_devices, input)
        assert isinstance(outputs, tuple)
        assert torch.allclose(input.to('mps'), outputs[0])

        target_devices = [0]
        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = Scatter.forward(target_devices, inputs)
        assert isinstance(outputs, tuple)
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.to('mps'), output[0])
