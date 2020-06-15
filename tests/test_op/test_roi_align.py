import os

import numpy as np
import torch

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
outputs = [([[[[1.0, 1.25], [1.5, 1.75]]]], [[[[3.0625, 0.4375],
                                               [0.4375, 0.0625]]]]),
           ([[[[1.0, 1.25], [1.5, 1.75]],
              [[4.0, 3.75], [3.5, 3.25]]]], [[[[3.0625, 0.4375],
                                               [0.4375, 0.0625]],
                                              [[3.0625, 0.4375],
                                               [0.4375, 0.0625]]]]),
           ([[[[1.9375, 4.75], [7.5625, 10.375]]]],
            [[[[0.47265625, 0.42968750, 0.42968750, 0.04296875],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.04296875, 0.03906250, 0.03906250, 0.00390625]]]])]


class TestRoiAlign(object):

    def test_roialign_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import RoIAlign
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0
        sampling_ratio = 2

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(np_input, device='cuda', requires_grad=True)
            rois = torch.tensor(np_rois, device='cuda')

            froipool = RoIAlign((pool_h, pool_w), spatial_scale,
                                sampling_ratio)

            if _USING_PARROTS:
                pass
                # gradcheck(froipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(froipool, (x, rois), eps=1e-2, atol=1e-2)

    def _test_roipool_allclose(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.op import roi_align
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
                np_input, dtype=dtype, device='cuda', requires_grad=True)
            rois = torch.tensor(np_rois, dtype=dtype, device='cuda')

            output = roi_align(x, rois, (pool_h, pool_w), spatial_scale,
                               sampling_ratio, 'avg', True)
            output.backward(torch.ones_like(output))
            assert np.allclose(
                output.data.type(torch.float).cpu().numpy(),
                np_output,
                atol=1e-3)
            assert np.allclose(
                x.grad.data.type(torch.float).cpu().numpy(),
                np_grad,
                atol=1e-3)

    def test_roipool_allclose(self):
        self._test_roipool_allclose(torch.float)
        self._test_roipool_allclose(torch.double)
        self._test_roipool_allclose(torch.half)
