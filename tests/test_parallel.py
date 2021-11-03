from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmcv.parallel import (MODULE_WRAPPERS, DataContainer, MMDataParallel,
                           MMDistributedDataParallel, collate,
                           is_module_wrapper)
from mmcv.parallel.distributed_deprecated import \
    MMDistributedDataParallel as DeprecatedMMDDP


def mock(*args, **kwargs):
    pass


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

    # _verify_model_across_ranks is added in torch1.9.0 so we should check
    # whether _verify_model_across_ranks is the member of torch.distributed
    # before mocking
    if hasattr(torch.distributed, '_verify_model_across_ranks'):
        torch.distributed._verify_model_across_ranks = mock

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
    class ModuleWrapper(object):

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    module_wraper = ModuleWrapper(model)
    assert is_module_wrapper(module_wraper)


def test_collate():
    batch = [{
        'img_metas':
        DataContainer(
            data={
                'flip': False,
                'flip_direction': 'horizontal'
            },
            stack=False,
            padding_value=0,
            cpu_only=True,
            pad_dims=None),
        'img':
        DataContainer(
            data=torch.rand(3, 4, 5),
            stack=True,
            padding_value=0,
            cpu_only=False,
            pad_dims=2),
        'gt_bboxes':
        DataContainer(
            data=torch.randint(0, 4, (2, 4)),
            stack=False,
            padding_value=0,
            cpu_only=False,
            pad_dims=None),
        'gt_labels':
        [torch.randint(0, 10, (1, )),
         torch.randint(0, 10, (1, ))],
        'filename':
        '/tmp/x1'
    }, {
        'img_metas':
        DataContainer(
            data={
                'flip': False,
                'flip_direction': 'horizontal'
            },
            stack=False,
            padding_value=0,
            cpu_only=False,
            pad_dims=None),
        'img':
        DataContainer(
            data=torch.rand(3, 6, 7),
            stack=True,
            padding_value=0,
            cpu_only=False,
            pad_dims=2),
        'gt_bboxes':
        DataContainer(
            data=torch.randint(0, 4, (2, 4)),
            stack=False,
            padding_value=0,
            cpu_only=False,
            pad_dims=None),
        'gt_labels':
        [torch.randint(0, 10, (1, )),
         torch.randint(0, 10, (1, ))],
        'filename':
        '/tmp/y1'
    }]
    result = collate(batch, samples_per_gpu=2)
    assert result['img'].data[0].shape == torch.Size([2, 3, 6, 7])
    assert len(result['img_metas'].data[0]) == 2
    assert len(result['gt_bboxes'].data[0]) == 2
    assert len(result['gt_labels']) == 2
    assert result['filename'] == ['/tmp/x1', '/tmp/y1']
    batch = [{
        'img_metas':
        DataContainer(
            data={
                'flip': False,
                'flip_direction': 'horizontal'
            },
            stack=False,
            padding_value=0,
            cpu_only=True,
            pad_dims=None),
        'img':
        DataContainer(
            data=torch.rand(3, 4, 5),
            stack=True,
            padding_value=0,
            cpu_only=False,
            pad_dims=3),
        'gt_bboxes':
        DataContainer(
            data=torch.randint(0, 4, (2, 4)),
            stack=False,
            padding_value=0,
            cpu_only=False,
            pad_dims=None),
        'gt_labels':
        [torch.randint(0, 10, (1, )),
         torch.randint(0, 10, (1, ))],
        'filename':
        '/tmp/z1'
    }]
    with pytest.raises(
            ValueError,
            match='pad_dims should be either None or less than dim'):
        collate(batch, samples_per_gpu=1)
