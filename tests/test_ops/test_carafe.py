# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE


class TestCarafe:

    def test_carafe_naive_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFENaive
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFENaive(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    def test_carafe_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFE
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support'))
    ])
    def test_carafe_allclose(self, device):
        try:
            from mmcv.ops import CARAFE
        except ModuleNotFoundError:
            pytest.skip('test requires compilation')

        np_feat = np.fromfile(
            'tests/data/for_carafe/carafe_feat.bin', dtype=np.float32)
        np_mask = np.fromfile(
            'tests/data/for_carafe/carafe_mask.bin', dtype=np.float32)
        np_output = np.fromfile(
            'tests/data/for_carafe/carafe_output.bin', dtype=np.float32)
        np_feat_grad = np.fromfile(
            'tests/data/for_carafe/carafe_feat_grad.bin', dtype=np.float32)
        np_mask_grad = np.fromfile(
            'tests/data/for_carafe/carafe_mask_grad.bin', dtype=np.float32)

        np_feat = np_feat.reshape((2, 64, 3, 3))
        np_mask = np_mask.reshape((2, 100, 6, 6))
        np_output = np_output.reshape((2, 64, 6, 6))
        np_feat_grad = np_feat_grad.reshape((2, 64, 3, 3))
        np_mask_grad = np_mask_grad.reshape((2, 100, 6, 6))

        feat = torch.tensor(
            np_feat, dtype=torch.float, device=device, requires_grad=True)
        mask = torch.tensor(
            np_mask, dtype=torch.float, device=device, requires_grad=True)

        carafe = CARAFE(5, 4, 2)

        output = carafe(feat, mask)
        output.backward(torch.ones_like(output))
        assert np.allclose(
            output.data.type(torch.float).cpu().numpy(), np_output, atol=1e-3)
        assert np.allclose(
            feat.grad.data.type(torch.float).cpu().numpy(),
            np_feat_grad,
            atol=1e-3)
        assert np.allclose(
            mask.grad.data.type(torch.float).cpu().numpy(),
            np_mask_grad,
            atol=1e-3)
