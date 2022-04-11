# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
import pytest
import torch
from mmcv.parallel.data_container import DataContainer
from mmcv.device.ipu import IS_IPU
if IS_IPU:
    from mmcv.device.ipu.model_converter import HierarchicalData

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU, reason='test case under ipu environment')


@skip_no_ipu
def test_HierarchicalData():
    # test complex data
    complex_data = {
        'a': torch.rand(3, 4),
        'b': np.random.rand(3, 4),
        'c': DataContainer({'a': torch.rand(3, 4), 'b': 4, 'c': 'd'}),
        'd': 123,
        'e': [1, 3, torch.rand(3, 4), np.random.rand(3, 4)],
        'f': {
            'a': torch.rand(3, 4),
            'b': np.random.rand(3, 4),
            'c': [1, 'asd']}}

    hd = HierarchicalData(logging.getLogger())
    hd.set_tree(complex_data)
    tensors = hd.get_tensors()
    tensors[0].add_(1)
    hd.set_tensors(tensors)
    tree = hd.tree
    tree['c'].data['a'].sub_(1)
    hd.set_tree(tree)
    tensors = hd.get_tensors()
    hd.quick()

    with pytest.raises(AssertionError,
                       match='original complex data is not torch.tensor'):
        hd.set_tree(torch.rand(3, 4))

    class AuxClass:
        pass
    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        hd.set_tree(AuxClass())

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        hd.set_tensors(tensors)

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        hd.get_tensors()

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        hd.clean_tensors()

    hd = HierarchicalData(logging.getLogger())
    hd.set_tree(complex_data)
    complex_data['a'] = torch.rand(3, 4)
    with pytest.raises(ValueError, match='all data except torch.Tensor'):
        new_complex_data = {**complex_data, 'b': np.random.rand(3, 4)}
        hd.update(new_complex_data)

    hd.update(new_complex_data, strict=False)

    hd.clean_tensors()

    # test single tensor
    single_tensor = torch.rand(3, 4)
    hd = HierarchicalData(logging.getLogger())
    hd.set_tree(single_tensor)
    hd.set_tree(torch.rand(3, 4))
    tensors = hd.get_tensors()
    hd.set_tensors(tensors)
