# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mmcv.parallel.data_container import DataContainer
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from mmcv.device.ipu import IPUDataLoader, cfg2options
    from mmcv.device.ipu.dataloader import collate

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU_AVAILABLE, reason='test case under ipu environment')


class ToyDataset(Dataset):

    def __getitem__(self, index):
        return 111

    def __len__(self, ):
        return 3


@skip_no_ipu
def test_ipu_dataloader():
    # test lazy initialization
    dataloader = IPUDataLoader(
        ToyDataset(), None, batch_size=256, num_workers=1, mode='async')
    options_cfg = {'train_cfg': {}, 'eval_cfg': {}}
    ipu_options = cfg2options(options_cfg)
    dataloader.init(ipu_options['training'])

    # test normal initialization
    options_cfg = {'train_cfg': {}, 'eval_cfg': {}}
    ipu_options = cfg2options(options_cfg)['training']
    dataloader = IPUDataLoader(
        ToyDataset(), ipu_options, batch_size=256, num_workers=1, mode='async')


@skip_no_ipu
def test_ipu_collate():
    with pytest.raises(TypeError, match='`batch` should be a sequence'):
        collate(123)

    with pytest.raises(TypeError, match='DataContainer is not supported'):
        collate([DataContainer(666)])

    data_list = [[1, 2, 3], [2, 3, 4], DataContainer(666)]
    batch0 = {
        'tensor': torch.rand(3, 4, 5),
        'arr': np.random.rand(3, 4, 5, 6),
        'data_list': data_list
    }
    batch1 = {
        'tensor': torch.rand(3, 4, 5),
        'arr': np.random.rand(3, 4, 5, 6),
        'data_list': data_list
    }
    batch = [batch1, batch0]
    results = collate(batch)
    assert results['tensor'].shape == (2, 3, 4, 5)
    assert results['arr'].shape == (2, 3, 4, 5, 6)
    for data in results['data_list']:
        for tensor in data:
            assert not isinstance(tensor, DataContainer)
            assert tensor.shape == (2, )
