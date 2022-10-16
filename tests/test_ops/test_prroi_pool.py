# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck

    _USING_PARROTS = False

inputs = [([[[[1., 2.], [3., 4.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2.], [3., 4.]], [[4., 3.], [2.,
                                               1.]]]], [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.], [9., 10., 13., 14.],
              [11., 12., 15., 16.]]]], [[0., 0., 0., 3., 3.]])]
outputs = [
    ([[[[1.75, 2.25], [2.75, 3.25]]]], [[[[1., 1.],
                                          [1., 1.]]]], [[0., 2., 4., 2., 4.]]),
    ([[[[1.75, 2.25], [2.75, 3.25]],
       [[3.25, 2.75], [2.25, 1.75]]]], [[[[1., 1.], [1., 1.]],
                                         [[1., 1.],
                                          [1., 1.]]]], [[0., 0., 0., 0., 0.]]),
    ([[[[3.75, 6.91666651],
        [10.08333302,
         13.25]]]], [[[[0.11111111, 0.22222224, 0.22222222, 0.11111111],
                       [0.22222224, 0.444444448, 0.44444448, 0.22222224],
                       [0.22222224, 0.44444448, 0.44444448, 0.22222224],
                       [0.11111111, 0.22222224, 0.22222224, 0.11111111]]]],
     [[0.0, 3.33333302, 6.66666603, 3.33333349, 6.66666698]])
]


class TestPrRoiPool:

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support'))
    ])
    def test_roipool_gradcheck(self, device):
        from mmcv.ops import PrRoIPool
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0

        for case in inputs:
            np_input = np.array(case[0], dtype=np.float32)
            np_rois = np.array(case[1], dtype=np.float32)

            x = torch.tensor(np_input, device=device, requires_grad=True)
            rois = torch.tensor(np_rois, device=device)

            froipool = PrRoIPool((pool_h, pool_w), spatial_scale)

            if _USING_PARROTS:
                gradcheck(froipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(froipool, (x, rois), eps=1e-2, atol=1e-2)

    def _test_roipool_allclose(self, device, dtype=torch.float):
        from mmcv.ops import prroi_pool
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0

        for case, output in zip(inputs, outputs):
            np_input = np.array(case[0], dtype=np.float32)
            np_rois = np.array(case[1], dtype=np.float32)
            np_output = np.array(output[0], dtype=np.float32)
            np_input_grad = np.array(output[1], dtype=np.float32)
            np_rois_grad = np.array(output[2], dtype=np.float32)

            x = torch.tensor(
                np_input, dtype=dtype, device=device, requires_grad=True)
            rois = torch.tensor(
                np_rois, dtype=dtype, device=device, requires_grad=True)

            output = prroi_pool(x, rois, (pool_h, pool_w), spatial_scale)
            output.backward(torch.ones_like(output))
            assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
            assert np.allclose(x.grad.data.cpu().numpy(), np_input_grad, 1e-3)
            assert np.allclose(rois.grad.data.cpu().numpy(), np_rois_grad,
                               1e-3)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support'))
    ])
    def test_roipool_allclose_float(self, device):
        self._test_roipool_allclose(device, dtype=torch.float)
