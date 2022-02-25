# Copyright (c) OpenMMLab. All rights reserved.
try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    from mmcv.transforms import ToTensor, to_tensor, ImageToTensor

import copy

import numpy as np
import pytest


@pytest.mark.skipif(condition=torch is None, reason='No torch in current env')
def test_to_tensor():

    # The type of the input object is torch.Tensor
    data_tensor = torch.tensor([1, 2, 3])
    tensor_from_tensor = to_tensor(data_tensor)
    assert isinstance(tensor_from_tensor, torch.Tensor)

    # The type of the input object is numpy.ndarray
    data_numpy = np.array([1, 2, 3])
    tensor_from_numpy = to_tensor(data_numpy)
    assert isinstance(tensor_from_numpy, torch.Tensor)

    # The type of the input object is list
    data_list = [1, 2, 3]
    tensor_from_list = to_tensor(data_list)
    assert isinstance(tensor_from_list, torch.Tensor)

    # The type of the input object is int
    data_int = 1
    tensor_from_int = to_tensor(data_int)
    assert isinstance(tensor_from_int, torch.Tensor)

    # The type of the input object is float
    data_float = 1.0
    tensor_from_float = to_tensor(data_float)
    assert isinstance(tensor_from_float, torch.Tensor)

    # The type of the input object is invalid
    with pytest.raises(TypeError):
        data_str = '123'
        _ = to_tensor(data_str)


@pytest.mark.skipif(condition=torch is None, reason='No torch in current env')
class TestToTensor:

    def test_init(self):
        TRANSFORM = ToTensor(keys=['img_label'])
        assert TRANSFORM.keys == ['img_label']

    def test_transform(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label', 'bbox'])

        # Test multi-level key and single-level key (multi-level key is
        # not in results)
        results = {'instances': {'label': [1]}, 'img_label': [1]}
        results_tensor = TRANSFORMS.transform(copy.deepcopy(results))
        assert isinstance(results_tensor['instances']['label'], list)
        assert isinstance(results_tensor['img_label'], torch.Tensor)

        # Test multi-level key (multi-level key is in results)
        results = {'instances': {'bbox': [[0, 0, 10, 10]]}, 'img_label': [1]}
        results_tensor = TRANSFORMS.transform(copy.deepcopy(results))
        assert isinstance(results_tensor['instances']['bbox'], torch.Tensor)

    def test_repr(self):
        TRANSFORMS = ToTensor(['instances.bbox', 'img_label'])
        TRANSFORMS_str = str(TRANSFORMS)
        isinstance(TRANSFORMS_str, str)


@pytest.mark.skipif(condition=torch is None, reason='No torch in current env')
class TestImageToTensor:

    def test_init(self):
        TRANSFORMS = ImageToTensor(['img'])
        assert TRANSFORMS.keys == ['img']

    def test_transform(self):
        TRANSFORMS = ImageToTensor(['img'])

        # image only has one channel
        results = {'img': np.zeros((224, 224))}
        results = TRANSFORMS.transform(results)
        assert results['img'].shape == (1, 224, 224)

        # image has three channels
        results = {'img': np.zeros((224, 224, 3))}
        results = TRANSFORMS.transform(results)
        assert results['img'].shape == (3, 224, 224)

    def test_repr(self):
        TRANSFORMS = ImageToTensor(['img'])
        TRANSFORMS_str = str(TRANSFORMS)
        assert isinstance(TRANSFORMS_str, str)