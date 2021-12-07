# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import (RoIAwarePool3d, points_in_boxes_all, points_in_boxes_cpu,
                      points_in_boxes_part)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_RoIAwarePool3d():
    roiaware_pool3d_max = RoIAwarePool3d(
        out_size=4, max_pts_per_voxel=128, mode='max')
    roiaware_pool3d_avg = RoIAwarePool3d(
        out_size=4, max_pts_per_voxel=128, mode='avg')
    rois = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 4.0, 6.0, -0.3 - np.pi / 2],
         [-10.0, 23.0, 16.0, 20.0, 10.0, 20.0, -0.5 - np.pi / 2]],
        dtype=torch.float32).cuda(
        )  # boxes (m, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=torch.float32).cuda()  # points (n, 3) in lidar coordinate
    pts_feature = pts.clone()

    pooled_features_max = roiaware_pool3d_max(
        rois=rois, pts=pts, pts_feature=pts_feature)
    assert pooled_features_max.shape == torch.Size([2, 4, 4, 4, 3])
    assert torch.allclose(pooled_features_max.sum(),
                          torch.tensor(51.100).cuda(), 1e-3)

    pooled_features_avg = roiaware_pool3d_avg(
        rois=rois, pts=pts, pts_feature=pts_feature)
    assert pooled_features_avg.shape == torch.Size([2, 4, 4, 4, 3])
    assert torch.allclose(pooled_features_avg.sum(),
                          torch.tensor(49.750).cuda(), 1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_points_in_boxes_part():
    boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3]],
         [[-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
        dtype=torch.float32).cuda(
        )  # boxes (b, t, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2]],
         [[3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9], [-21.3, -52, -5],
          [0, 0, 0], [6, 7, 8], [-2, -3, -4], [6, 4, 9]]],
        dtype=torch.float32).cuda()  # points (b, m, 3) in lidar coordinate

    point_indices = points_in_boxes_part(points=pts, boxes=boxes)
    expected_point_indices = torch.tensor(
        [[0, 0, 0, 0, 0, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([2, 8])
    assert (point_indices == expected_point_indices).all()

    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 0.523598]]],
                         dtype=torch.float32).cuda()  # 30 degrees
    pts = torch.tensor(
        [[[4, 6.928, 0], [6.928, 4, 0], [4, -6.928, 0], [6.928, -4, 0],
          [-4, 6.928, 0], [-6.928, 4, 0], [-4, -6.928, 0], [-6.928, -4, 0]]],
        dtype=torch.float32).cuda()
    point_indices = points_in_boxes_part(points=pts, boxes=boxes)
    expected_point_indices = torch.tensor([[-1, -1, 0, -1, 0, -1, -1, -1]],
                                          dtype=torch.int32).cuda()
    assert (point_indices == expected_point_indices).all()


def test_points_in_boxes_cpu():
    boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
          [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
        dtype=torch.float32
    )  # boxes (m, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [
              -16, -18, 9
          ], [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]]],
        dtype=torch.float32)  # points (n, 3) in lidar coordinate

    point_indices = points_in_boxes_cpu(points=pts, boxes=boxes)
    expected_point_indices = torch.tensor(
        [[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0],
          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.int32)
    assert point_indices.shape == torch.Size([1, 15, 2])
    assert (point_indices == expected_point_indices).all()

    boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 0.523598]]],
                         dtype=torch.float32)  # 30 degrees
    pts = torch.tensor(
        [[[4, 6.928, 0], [6.928, 4, 0], [4, -6.928, 0], [6.928, -4, 0],
          [-4, 6.928, 0], [-6.928, 4, 0], [-4, -6.928, 0], [-6.928, -4, 0]]],
        dtype=torch.float32)
    point_indices = points_in_boxes_cpu(points=pts, boxes=boxes)
    expected_point_indices = torch.tensor(
        [[[0], [0], [1], [0], [1], [0], [0], [0]]], dtype=torch.int32)
    assert (point_indices == expected_point_indices).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_points_in_boxes_all():

    boxes = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.3],
          [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]]],
        dtype=torch.float32).cuda(
        )  # boxes (m, 7) with bottom center in lidar coordinate
    pts = torch.tensor(
        [[[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
          [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
          [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [
              -16, -18, 9
          ], [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]]],
        dtype=torch.float32).cuda()  # points (n, 3) in lidar coordinate

    point_indices = points_in_boxes_all(points=pts, boxes=boxes)
    expected_point_indices = torch.tensor(
        [[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, 0], [0, 0],
          [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]],
        dtype=torch.int32).cuda()
    assert point_indices.shape == torch.Size([1, 15, 2])
    assert (point_indices == expected_point_indices).all()
