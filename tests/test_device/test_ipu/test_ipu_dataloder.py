# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mmcv.device.ipu import IS_IPU
from mmcv.parallel.data_container import DataContainer

if IS_IPU:
    from mmcv.device.ipu import IPUDataLoader, cast_to_options
    from mmcv.device.ipu.dataloader import collate

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU, reason='test case under ipu environment')


class ToyDataset(Dataset):

    def __getitem__(self, index):
        return 111

    def __len__(self, ):
        return 3


@skip_no_ipu
def test_ipu_dataloader():
    # test lazy initialization
    dataloader = IPUDataLoader(
        None, ToyDataset(), batch_size=256, num_workers=1, mode='async')
    options_cfg = {'train_cfg': {}, 'eval_cfg': {}}
    ipu_options = cast_to_options(options_cfg)
    dataloader.init(ipu_options['training'])

    # test normal initialization
    options_cfg = {'train_cfg': {}, 'eval_cfg': {}}
    ipu_options = cast_to_options(options_cfg)['training']
    dataloader = IPUDataLoader(
        ipu_options, ToyDataset(), batch_size=256, num_workers=1, mode='async')


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
    tensor_shape = torch.Tensor(list(results['tensor'].shape))
    arr_shape = torch.Tensor(list(results['arr'].shape))
    assert torch.equal(torch.Tensor([2, 3, 4, 5]), tensor_shape)
    assert torch.equal(torch.Tensor([2, 3, 4, 5, 6]), arr_shape)
    for data in results['data_list']:
        for tensor in data:
            assert not isinstance(tensor, DataContainer)
            data_in_list_shape = torch.Tensor(list(tensor.shape))
            assert torch.equal(torch.Tensor([2]), data_in_list_shape)
