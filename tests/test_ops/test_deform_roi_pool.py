# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck
    _USING_PARROTS = False

cur_dir = os.path.dirname(os.path.abspath(__file__))

inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2.], [3., 4.]], [[4., 3.], [2.,
                                               1.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
              [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]
outputs = [([[[[1, 1.25], [1.5, 1.75]]]], [[[[3.0625, 0.4375],
                                             [0.4375, 0.0625]]]]),
           ([[[[1., 1.25], [1.5, 1.75]], [[4, 3.75],
                                          [3.5, 3.25]]]], [[[[3.0625, 0.4375],
                                                             [0.4375, 0.0625]],
                                                            [[3.0625, 0.4375],
                                                             [0.4375,
                                                              0.0625]]]]),
           ([[[[1.9375, 4.75],
               [7.5625,
                10.375]]]], [[[[0.47265625, 0.4296875, 0.4296875, 0.04296875],
                               [0.4296875, 0.390625, 0.390625, 0.0390625],
                               [0.4296875, 0.390625, 0.390625, 0.0390625],
                               [0.04296875, 0.0390625, 0.0390625,
                                0.00390625]]]])]


class TestDeformRoIPool:

    def test_deform_roi_pool_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import DeformRoIPoolPack
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, device='cuda', dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device='cuda', dtype=torch.float)
            output_c = x.size(1)

            droipool = DeformRoIPoolPack((pool_h, pool_w),
                                         output_c,
                                         spatial_scale=spatial_scale,
                                         sampling_ratio=sampling_ratio).cuda()

            if _USING_PARROTS:
                gradcheck(droipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(droipool, (x, rois), eps=1e-2, atol=1e-2)

    def test_modulated_deform_roi_pool_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import ModulatedDeformRoIPoolPack
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(
                np_input, device='cuda', dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device='cuda', dtype=torch.float)
            output_c = x.size(1)

            droipool = ModulatedDeformRoIPoolPack(
                (pool_h, pool_w),
                output_c,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio).cuda()

            if _USING_PARROTS:
                gradcheck(droipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(droipool, (x, rois), eps=1e-2, atol=1e-2)

    def _test_deform_roi_pool_allclose(self, device, dtype=torch.float):
        from mmcv.ops import DeformRoIPoolPack
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case, output in zip(inputs, outputs):
            np_input = np.array(case[0])
            np_rois = np.array(case[1])
            np_output = np.array(output[0])
            np_grad = np.array(output[1])

            x = torch.tensor(
                np_input, device=device, dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device=device, dtype=torch.float)
            output_c = x.size(1)
            droipool = DeformRoIPoolPack(
                (pool_h, pool_w),
                output_c,
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio).to(device)

            output = droipool(x, rois)
            output.backward(torch.ones_like(output))
            assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
            assert np.allclose(x.grad.data.cpu().numpy(), np_grad, 1e-3)

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
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support'))
    ])
    @pytest.mark.parametrize('dtype', [
        torch.float,
        pytest.param(
            torch.double,
            marks=pytest.mark.skipif(
                IS_MLU_AVAILABLE,
                reason='MLU does not support for 64-bit floating point')),
        torch.half
    ])
    def test_deform_roi_pool_allclose(self, device, dtype):
        self._test_deform_roi_pool_allclose(device, dtype)
