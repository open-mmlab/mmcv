# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import RoIPointPool3d


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_roipoint():
    feats = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=torch.float32).unsqueeze(0).cuda()
    points = feats.clone()
    rois = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
                          [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
                        dtype=torch.float32).cuda()

    roipoint_pool3d = RoIPointPool3d(num_sampled_points=4)
    roi_feat, empty_flag = roipoint_pool3d(feats, points, rois)
    expected_roi_feat = torch.tensor([[[[1, 2, 3.3, 1, 2, 3.3],
                                        [1.2, 2.5, 3, 1.2, 2.5, 3],
                                        [0.8, 2.1, 3.5, 0.8, 2.1, 3.5],
                                        [1.6, 2.6, 3.6, 1.6, 2.6, 3.6]],
                                       [[-9.2, 21, 18.2, -9.2, 21, 18.2],
                                        [-9.2, 21, 18.2, -9.2, 21, 18.2],
                                        [-9.2, 21, 18.2, -9.2, 21, 18.2],
                                        [-9.2, 21, 18.2, -9.2, 21,
                                         18.2]]]]).cuda()
    expected_empty_flag = torch.tensor([[0, 0]]).int().cuda()

    assert torch.allclose(roi_feat, expected_roi_feat)
    assert torch.allclose(empty_flag, expected_empty_flag)
