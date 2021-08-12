import copy
import os.path as osp

import numpy as np
import torch

from mmcv.datasets import PIPELINES


class TestFormatting:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), '../../data')

    def test_to_tensor(self):
        # test all convert types
        transform_cfg = dict(type='ToTensor', keys=['label'])
        transform = PIPELINES.build(transform_cfg)
        # Tensor
        data_info = dict(label=torch.tensor([1, 2]))
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['label'], torch.Tensor)
        assert torch.allclose(results['label'], torch.tensor([1, 2]))

        # np.ndarray
        data_info = dict(label=np.array([1, 2]).astype(np.float32))
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['label'], torch.Tensor)
        assert torch.allclose(results['label'], torch.FloatTensor([1., 2.]))

        # int
        data_info = dict(label=1)
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['label'], torch.LongTensor)
        assert torch.allclose(results['label'], torch.tensor([1]))

        # float
        data_info = dict(label=1.)
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['label'], torch.FloatTensor)
        assert torch.allclose(results['label'], torch.tensor([1.]))

        # tuple or list
        transform_cfg = dict(type='ToTensor', keys=['label1', 'label2'])
        transform = PIPELINES.build(transform_cfg)
        data_info = dict(label1=[1, 2], label2=(1., 2.))
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['label1'], torch.Tensor)
        assert torch.allclose(results['label1'], torch.tensor([1, 2]))
        assert isinstance(results['label2'], torch.Tensor)
        assert torch.allclose(results['label2'], torch.tensor([1., 2.]))

        # test nested keys
        transform_cfg = dict(type='ToTensor', keys=['img', 'ann.label'])
        transform = PIPELINES.build(transform_cfg)
        data_info = dict(img=[1, 2], ann=dict(label=np.array([1, 2])))
        results = transform(copy.deepcopy(data_info))
        assert isinstance(results['img'], torch.Tensor)
        assert torch.allclose(results['img'], torch.tensor([1, 2]))
        assert isinstance(results['ann']['label'], torch.Tensor)
        assert torch.allclose(results['ann']['label'], torch.tensor([1, 2]))
