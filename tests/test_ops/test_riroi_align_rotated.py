# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import RiRoIAlignRotated

if torch.__version__ == 'parrots':
    from parrots.autograd import gradcheck
    _USING_PARROTS = True
else:
    from torch.autograd import gradcheck
    _USING_PARROTS = False

np_feature = np.array([[[[1, 2], [3, 4]], [[1, 2], [4, 3]], [[4, 3], [2, 1]],
                        [[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13,
                                                                       14]],
                        [[11, 12], [15, 16]], [[1, 1], [2, 2]]]])
np_rois = np.array([[0., 0.5, 0.5, 1., 1., np.pi / 3],
                    [0., 1., 1., 3., 3., np.pi / 2]])
expect_output = np.array([[[[1.8425, 1.3516], [2.3151, 1.8241]],
                           [[2.4779, 1.7416], [3.2173, 2.5632]],
                           [[2.7149, 2.2638], [2.6540, 2.3673]],
                           [[2.9461, 2.8638], [2.8028, 2.7205]],
                           [[4.1943, 2.7214], [5.6119, 4.1391]],
                           [[7.5276, 6.0547], [8.9453, 7.4724]],
                           [[12.1943, 10.7214], [13.6119, 12.1391]],
                           [[9.5489, 8.4237], [10.5763, 9.4511]]],
                          [[[7.6562, 12.5625], [4.0000, 6.6250]],
                           [[1.0000, 1.3125], [0.5000, 0.6562]],
                           [[1.6562, 1.9375], [1.0000, 1.3125]],
                           [[1.8438, 2.0547], [0.7500, 1.1562]],
                           [[0.8438, 3.0625], [0.2500, 1.1875]],
                           [[2.6562, 2.5625], [1.5000, 1.6250]],
                           [[3.6562, 4.5625], [2.0000, 2.6250]],
                           [[6.6562, 10.5625], [3.5000, 5.6250]]]])

expect_grad = np.array([[[[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]],
                         [[1.4727, 1.5586], [1.5586, 1.6602]]]])

pool_h = 2
pool_w = 2
spatial_scale = 1.0
num_samples = 2
sampling_ratio = 2
num_orientations = 8
clockwise = False


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_roialign_rotated_gradcheck():
    x = torch.tensor(
        np_feature, dtype=torch.float, device='cuda', requires_grad=True)
    rois = torch.tensor(np_rois, dtype=torch.float, device='cuda')
    froipool = RiRoIAlignRotated((pool_h, pool_w), spatial_scale, num_samples,
                                 num_orientations, clockwise)
    if _USING_PARROTS:
        gradcheck(
            froipool, (x, rois), no_grads=[rois], delta=1e-3, pt_atol=1e-3)
    else:
        gradcheck(froipool, (x, rois), eps=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_roialign_rotated_allclose():
    x = torch.tensor(
        np_feature, dtype=torch.float, device='cuda', requires_grad=True)
    rois = torch.tensor(np_rois, dtype=torch.float, device='cuda')
    froipool = RiRoIAlignRotated((pool_h, pool_w), spatial_scale, num_samples,
                                 num_orientations, clockwise)
    output = froipool(x, rois)
    output.backward(torch.ones_like(output))
    assert np.allclose(
        output.data.type(torch.float).cpu().numpy(), expect_output, atol=1e-3)
    assert np.allclose(
        x.grad.data.type(torch.float).cpu().numpy(), expect_grad, atol=1e-3)
