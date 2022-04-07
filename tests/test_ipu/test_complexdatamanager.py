# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
import pytest
import torch
from mmcv.parallel.data_container import DataContainer
from mmcv.device.ipu import IPU_MODE
if IPU_MODE:
    from mmcv.device.ipu.model_converter import ComplexDataManager

skip_no_ipu = pytest.mark.skipif(
    not IPU_MODE, reason='test case under ipu environment')


@skip_no_ipu
def test_complexdatamanager():
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

    cdm = ComplexDataManager(logging.getLogger())
    cdm.set_tree(complex_data)
    tensors = cdm.get_tensors()
    tensors[0].add_(1)
    cdm.set_tensors(tensors)
    tree = cdm.tree
    tree['c'].data['a'].sub_(1)
    cdm.set_tree(tree)
    tensors = cdm.get_tensors()
    cdm.quick()

    with pytest.raises(AssertionError,
                       match='original complex data is not torch.tensor'):
        cdm.set_tree(torch.rand(3, 4))

    class AuxClass:
        pass
    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        cdm.set_tree(AuxClass())

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        cdm.set_tensors(tensors)

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        cdm.get_tensors()

    with pytest.raises(NotImplementedError, match='not supported datatype:'):
        complex_data['a'] = AuxClass()
        cdm.clean_tensors()

    cdm = ComplexDataManager(logging.getLogger())
    cdm.set_tree(complex_data)
    complex_data['a'] = torch.rand(3, 4)
    with pytest.raises(ValueError, match='all data except torch.Tensor'):
        new_complex_data = {**complex_data, 'b': np.random.rand(3, 4)}
        cdm.update(new_complex_data)

    cdm.update(new_complex_data, strict=False)

    cdm.clean_tensors()

    # test single tensor
    single_tensor = torch.rand(3, 4)
    cdm = ComplexDataManager(logging.getLogger())
    cdm.set_tree(single_tensor)
    cdm.set_tree(torch.rand(3, 4))
    tensors = cdm.get_tensors()
    cdm.set_tensors(tensors)
