# Copyright (c) OpenMMLab. All rights reserved.
import logging

import numpy as np
import pytest
import torch

from mmcv.parallel.data_container import DataContainer
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from mmcv.device.ipu.hierarchical_data_manager import \
        HierarchicalDataManager

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU_AVAILABLE, reason='test case under ipu environment')


@skip_no_ipu
def test_HierarchicalData():
    # test hierarchical data
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
    all_tensors = []
    all_tensors.append(hierarchical_data_sample['a'])
    all_tensors.append(hierarchical_data_sample['c'].data['a'])
    all_tensors.append(hierarchical_data_sample['e'][2])
    all_tensors.append(hierarchical_data_sample['f']['a'])
    all_tensors_id = [id(ele) for ele in all_tensors]

    hd = HierarchicalDataManager(logging.getLogger())
    hd.record_hierarchical_data(hierarchical_data_sample)
    tensors = hd.collect_all_tensors()
    for t in tensors:
        assert id(t) in all_tensors_id
    tensors[0].add_(1)
    hd.update_all_tensors(tensors)
    data = hd.hierarchical_data
    data['c'].data['a'].sub_(1)
    hd.record_hierarchical_data(data)
    tensors = hd.collect_all_tensors()
    for t in tensors:
        assert id(t) in all_tensors_id
    hd.quick()

    with pytest.raises(
            AssertionError,
            match='original hierarchical data is not torch.tensor'):
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
        hd.collect_all_tensors()

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
        hd.update_hierarchical_data(new_hierarchical_data_sample)

    hd.update_hierarchical_data(new_hierarchical_data_sample, strict=False)

    hd.clean_all_tensors()

    # test single tensor
    single_tensor = torch.rand(3, 4)
    hd = HierarchicalDataManager(logging.getLogger())
    hd.record_hierarchical_data(single_tensor)
    tensors = hd.collect_all_tensors()
    assert len(tensors) == 1 and single_tensor in tensors
    single_tensor_to_update = [torch.rand(3, 4)]
    hd.update_all_tensors(single_tensor_to_update)
    new_tensors = hd.collect_all_tensors()
    assert new_tensors == single_tensor_to_update
