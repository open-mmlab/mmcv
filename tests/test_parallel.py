# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.parallel import (MODULE_WRAPPERS, MMDataParallel,
                           MMDistributedDataParallel, is_module_wrapper)
from mmcv.parallel._functions import Scatter, get_input_device, scatter
from mmcv.parallel.distributed_deprecated import \
    MMDistributedDataParallel as DeprecatedMMDDP
from mmcv.utils import Registry


def mock(*args, **kwargs):
    pass


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not supported in parrots now')
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

    # _verify_model_across_ranks is added in torch1.9.0,
    # _verify_params_across_processes is added in torch1.11.0,
    # so we should check whether _verify_model_across_ranks
    # and _verify_params_across_processes are the member of
    # torch.distributed before mocking
    if hasattr(torch.distributed, '_verify_model_across_ranks'):
        torch.distributed._verify_model_across_ranks = mock
    if hasattr(torch.distributed, '_verify_params_across_processes'):
        torch.distributed._verify_params_across_processes = mock

    model = Model()
    assert not is_module_wrapper(model)

    dp = DataParallel(model)
    assert is_module_wrapper(dp)

    mmdp = MMDataParallel(model)
    assert is_module_wrapper(mmdp)

    ddp = DistributedDataParallel(model, process_group=MagicMock())
    assert is_module_wrapper(ddp)

    mmddp = MMDistributedDataParallel(model, process_group=MagicMock())
    assert is_module_wrapper(mmddp)

    deprecated_mmddp = DeprecatedMMDDP(model)
    assert is_module_wrapper(deprecated_mmddp)

    # test module wrapper registry
    @MODULE_WRAPPERS.register_module()
    class ModuleWrapper:

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    module_wraper = ModuleWrapper(model)
    assert is_module_wrapper(module_wraper)

    # test module wrapper registry in downstream repo
    MMRAZOR_MODULE_WRAPPERS = Registry(
        'mmrazor module wrapper', parent=MODULE_WRAPPERS, scope='mmrazor')
    MMPOSE_MODULE_WRAPPERS = Registry(
        'mmpose module wrapper', parent=MODULE_WRAPPERS, scope='mmpose')

    @MMRAZOR_MODULE_WRAPPERS.register_module()
    class ModuleWrapperInRazor:

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    @MMPOSE_MODULE_WRAPPERS.register_module()
    class ModuleWrapperInPose:

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    wrapped_module = ModuleWrapperInRazor(model)
    assert is_module_wrapper(wrapped_module)

    wrapped_module = ModuleWrapperInPose(model)
    assert is_module_wrapper(wrapped_module)


def test_get_input_device():
    # if the device is CPU, return -1
    input = torch.zeros([1, 3, 3, 3])
    assert get_input_device(input) == -1
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    assert get_input_device(inputs) == -1

    # if the device is GPU, return the index of device
    if torch.cuda.is_available():
        input = torch.zeros([1, 3, 3, 3]).cuda()
        assert get_input_device(input) == 0
        inputs = [
            torch.zeros([1, 3, 3, 3]).cuda(),
            torch.zeros([1, 4, 4, 4]).cuda()
        ]
        assert get_input_device(inputs) == 0

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

    # if the device is GPU, copy the input from CPU to GPU
    if torch.cuda.is_available():
        input = torch.zeros([1, 3, 3, 3])
        output = scatter(input=input, devices=[0])
        assert torch.allclose(input.cuda(), output)

        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = scatter(input=inputs, devices=[0])
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.cuda(), output)

    # input should be a tensor or list of tensor
    with pytest.raises(Exception):
        scatter(5, [-1])


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not supported in parrots now')
def test_Scatter():
    # if the device is CPU, just return the input
    target_gpus = [-1]
    input = torch.zeros([1, 3, 3, 3])
    outputs = Scatter.forward(target_gpus, input)
    assert isinstance(outputs, tuple)
    assert torch.allclose(input, outputs[0])

    target_gpus = [-1]
    inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
    outputs = Scatter.forward(target_gpus, inputs)
    assert isinstance(outputs, tuple)
    for input, output in zip(inputs, outputs):
        assert torch.allclose(input, output)

    # if the device is GPU, copy the input from CPU to GPU
    if torch.cuda.is_available():
        target_gpus = [0]
        input = torch.zeros([1, 3, 3, 3])
        outputs = Scatter.forward(target_gpus, input)
        assert isinstance(outputs, tuple)
        assert torch.allclose(input.cuda(), outputs[0])

        target_gpus = [0]
        inputs = [torch.zeros([1, 3, 3, 3]), torch.zeros([1, 4, 4, 4])]
        outputs = Scatter.forward(target_gpus, inputs)
        assert isinstance(outputs, tuple)
        for input, output in zip(inputs, outputs):
            assert torch.allclose(input.cuda(), output[0])
