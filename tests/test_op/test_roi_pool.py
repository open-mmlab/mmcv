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
outputs = [([[[[1., 2.], [3., 4.]]]], [[[[1., 1.], [1., 1.]]]]),
           ([[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]]], [[[[1., 1.],
                                                               [1., 1.]],
                                                              [[1., 1.],
                                                               [1., 1.]]]]),
           ([[[[4., 8.], [12., 16.]]]], [[[[0., 0., 0., 0.], [0., 1., 0., 1.],
                                           [0., 0., 0., 0.], [0., 1., 0.,
                                                              1.]]]])]


class TestRoiPool(object):

    def test_roipool_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import RoIPool
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0

        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])

            x = torch.tensor(np_input, device='cuda', requires_grad=True)
            rois = torch.tensor(np_rois, device='cuda')

            froipool = RoIPool((pool_h, pool_w), spatial_scale)

            if _USING_PARROTS:
                pass
                # gradcheck(froipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(froipool, (x, rois), eps=1e-2, atol=1e-2)

    def _test_roipool_allclose(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.op import roi_pool
        pool_h = 2
        pool_w = 2
        spatial_scale = 1.0

        for case, output in zip(inputs, outputs):
            np_input = np.array(case[0])
            np_rois = np.array(case[1])
            np_output = np.array(output[0])
            np_grad = np.array(output[1])

            x = torch.tensor(
                np_input, dtype=dtype, device='cuda', requires_grad=True)
            rois = torch.tensor(np_rois, dtype=dtype, device='cuda')

            output = roi_pool(x, rois, (pool_h, pool_w), spatial_scale)
            output.backward(torch.ones_like(output))
            assert np.allclose(output.data.cpu().numpy(), np_output, 1e-3)
            assert np.allclose(x.grad.data.cpu().numpy(), np_grad, 1e-3)

    def test_roipool_allclose(self):
        self._test_roipool_allclose(torch.double)
        self._test_roipool_allclose(torch.float)
        self._test_roipool_allclose(torch.half)
