# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MUSA_AVAILABLE


class TestCarafe:

    def test_carafe_naive_gradcheck(self):
        if (not torch.cuda.is_available()) and (not IS_MUSA_AVAILABLE):
            return
        from mmcv.ops import CARAFENaive
        if IS_CUDA_AVAILABLE:
            feat = torch.randn(
                2, 64, 3, 3, requires_grad=True, device='cuda').double()
            mask = torch.randn(
                2, 100, 6, 6, requires_grad=True,
                device='cuda').sigmoid().double()
            gradcheck(CARAFENaive(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    def test_carafe_gradcheck(self):
        if (not torch.cuda.is_available()) and (not IS_MUSA_AVAILABLE):
            return
        from mmcv.ops import CARAFE
        if IS_CUDA_AVAILABLE:
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
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'musa',
            marks=pytest.mark.skipif(
                not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
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

        # feat = torch.tensor(
        #     np_feat, dtype=torch.float, device=device, requires_grad=True)
        # mask = torch.tensor(
        #     np_mask, dtype=torch.float, device=device, requires_grad=True)

        # feat = torch.tensor(
        #     np_feat, dtype=torch.float, requires_grad=True).to(device)
        # mask = torch.tensor(
        #     np_mask, dtype=torch.float, requires_grad=True).to(device)
        # feat = torch.tensor(
        #     np_feat, dtype=torch.float).to(device)
        # mask = torch.tensor(
        #     np_mask, dtype=torch.float).to(device)
        # feat_cpu = torch.from_numpy(np_feat).to(torch.float)
        # mask_cpu = torch.from_numpy(np_mask).to(torch.float)

        # if device == 'musa':
        #     feat =feat_cpu.musa()
        #     mask =mask_cpu.musa()
        # else:
        #     feat =feat_cpu.to(device)
        #     mask =mask_cpu.to(device)
        # feat.requires_grad = True
        # mask.requires_grad = True
        feat_cpu = torch.FloatTensor(np_feat)
        mask_cpu = torch.FloatTensor(np_mask)
        feat = feat_cpu.to(device)
        mask = mask_cpu.to(device)
        feat.requires_grad = True
        mask.requires_grad = True
        # pytest.set_trace()

        carafe = CARAFE(5, 4, 2)
        carafe.to(device)
        carafe.train()
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
