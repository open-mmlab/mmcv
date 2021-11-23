import os

import numpy as np
import pytest
import torch

from mmcv.ops import DeformRoIPoolPack, ModulatedDeformRoIPoolPack

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


class TestDeformRoIPool(object):

    def setup_class(self):
        self.pool_h = 2
        self.pool_w = 2
        self.spatial_scale = 1.0
        self.sampling_ratio = 2
        self._DeformRoIPoolPack = {
            'plain': DeformRoIPoolPack,
            'modulated': ModulatedDeformRoIPoolPack
        }

    @pytest.mark.parametrize('mode', ['plain', 'modulated'])
    def test_deform_roi_pool_gradcheck(self, mode):
        if not torch.cuda.is_available():
            return
        for case in inputs:
            np_input = np.array(case[0])
            np_rois = np.array(case[1])
            x = torch.tensor(
                np_input, device='cuda', dtype=torch.float, requires_grad=True)
            rois = torch.tensor(np_rois, device='cuda', dtype=torch.float)
            output_c = x.size(1)

            droipool = self._DeformRoIPoolPack[mode](
                (self.pool_h, self.pool_w),
                output_c,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio).cuda()

            if _USING_PARROTS:
                gradcheck(droipool, (x, rois), no_grads=[rois])
            else:
                gradcheck(droipool, (x, rois), eps=1e-2, atol=1e-2)
