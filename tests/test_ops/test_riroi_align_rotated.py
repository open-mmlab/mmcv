import numpy as np
import pytest
import torch
from torch.autograd import gradcheck

from mmcv.ops import RiRoIAlignRotated

np_feature = np.array([[[[1, 2], [3, 4]], [[1, 2], [4, 3]], [[4, 3], [2, 1]],
                        [[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13,
                                                                       14]],
                        [[11, 12], [15, 16]], [[1, 1], [2, 2]]]])
np_rois = np.array([[0., 0.5, 0.5, 1., 1., np.pi / 3],
                    [0., 1., 1., 3., 3., np.pi / 2]])
expect_output = np.array([[[[1.4126, 2.0682], [1.5985, 2.2541]],
                           [[1.9047, 2.8881], [2.0708, 3.1364]],
                           [[2.2850, 2.4071], [2.5107, 2.7972]],
                           [[2.8028, 2.7205], [2.9461, 2.8638]],
                           [[2.9044, 4.8711], [3.4622, 5.4289]],
                           [[6.2377, 8.2045], [6.7955, 8.7623]],
                           [[10.9044, 12.8711], [11.4622, 13.4289]],
                           [[8.5457, 10.0001], [8.9999, 10.4543]]],
                          [[[6.6250, 4.0000], [12.5625, 7.6562]],
                           [[0.6562, 0.5000], [1.3125, 1.0000]],
                           [[1.3125, 1.0000], [1.9375, 1.6562]],
                           [[1.1562, 0.7500], [2.0547, 1.8438]],
                           [[1.1875, 0.2500], [3.0625, 0.8438]],
                           [[1.6250, 1.5000], [2.5625, 2.6562]],
                           [[2.6250, 2.0000], [4.5625, 3.6562]],
                           [[5.6250, 3.5000], [10.5625, 6.6562]]]])

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
sample_num = 2
sampling_ratio = 2
nOrientation = 8
clockwise = True


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_roialign_rotated_gradcheck():
    x = torch.tensor(
        np_feature, dtype=torch.float, device='cuda', requires_grad=True)
    rois = torch.tensor(np_rois, dtype=torch.float, device='cuda')
    froipool = RiRoIAlignRotated((pool_h, pool_w), spatial_scale, sample_num,
                                 nOrientation, clockwise)
    gradcheck(froipool, (x, rois), eps=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_roialign_rotated_allclose():
    x = torch.tensor(
        np_feature, dtype=torch.float, device='cuda', requires_grad=True)
    rois = torch.tensor(np_rois, dtype=torch.float, device='cuda')
    froipool = RiRoIAlignRotated((pool_h, pool_w), spatial_scale, sample_num,
                                 nOrientation, clockwise)
    output = froipool(x, rois)
    output.backward(torch.ones_like(output))
    assert np.allclose(
        output.data.type(torch.float).cpu().numpy(), expect_output, atol=1e-3)
    assert np.allclose(
        x.grad.data.type(torch.float).cpu().numpy(), expect_grad, atol=1e-3)
