import logging
import numpy as np
import pytest
import torch

from mmcv.parallel.data_container import DataContainer
from mmcv.runner.ipu.model_converter import ComplexDataManager

def test_complexdatamanager():
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
    tree = cdm.get_tree()
    tree['c'].data['a'].sub_(1)
    cdm.set_tree(tree)
    tensors = cdm.get_tensors()
    cdm.logger_off()
