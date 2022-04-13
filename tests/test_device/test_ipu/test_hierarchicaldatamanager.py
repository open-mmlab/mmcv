# Copyright (c) OpenMMLab. All rights reserved.
import logging

import numpy as np
import pytest
import torch

from mmcv.device.ipu import IS_IPU
from mmcv.parallel.data_container import DataContainer

if IS_IPU:
    from mmcv.device.ipu.hierarchical_data_manager import \
        HierarchicalDataManager

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU, reason='test case under ipu environment')


@skip_no_ipu
def test_HierarchicalData():
    # test complex data
    hierarchical_data_sample = {
        'a': torch.rand(3, 4),
        'b': np.random.rand(3, 4),
        'c': DataContainer({
            'a': torch.rand(3, 4),
            'b': 4,
            'c': 'd'
        }),
        'd': 123,
        'e': [1, 3, torch.rand(3, 4),
              np.random.rand(3, 4)],
        'f': {
            'a': torch.rand(3, 4),
            'b': np.random.rand(3, 4),
            'c': [1, 'asd']
        }
    }

    hd = HierarchicalDataManager(logging.getLogger())
    hd.record_hierarchical_data(hierarchical_data_sample)
    tensors = hd.get_all_tensors()
    tensors[0].add_(1)
    hd.update_all_tensors(tensors)
    data = hd.data
    data['c'].data['a'].sub_(1)
    hd.record_hierarchical_data(data)
    tensors = hd.get_all_tensors()
    hd.quick()

    with pytest.raises(
            AssertionError, match='original complex data is not torch.tensor'):
        hd.record_hierarchical_data(torch.rand(3, 4))

    class AuxClass:
        pass

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        hd.record_hierarchical_data(AuxClass())

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        hierarchical_data_sample['a'] = AuxClass()
        hd.update_all_tensors(tensors)

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        hierarchical_data_sample['a'] = AuxClass()
        hd.get_all_tensors()

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        hierarchical_data_sample['a'] = AuxClass()
        hd.clean_all_tensors()

    hd = HierarchicalDataManager(logging.getLogger())
    hd.record_hierarchical_data(hierarchical_data_sample)
    hierarchical_data_sample['a'] = torch.rand(3, 4)
    with pytest.raises(ValueError, match='all data except torch.Tensor'):
        new_hierarchical_data_sample = {
            **hierarchical_data_sample, 'b': np.random.rand(3, 4)
        }
        hd.update(new_hierarchical_data_sample)

    hd.update(new_hierarchical_data_sample, strict=False)

    hd.clean_all_tensors()

    # test single tensor
    single_tensor = torch.rand(3, 4)
    hd = HierarchicalDataManager(logging.getLogger())
    hd.record_hierarchical_data(single_tensor)
    hd.record_hierarchical_data(torch.rand(3, 4))
    tensors = hd.get_all_tensors()
    hd.update_all_tensors(tensors)
