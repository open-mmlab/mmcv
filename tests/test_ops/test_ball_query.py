# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import ball_query
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'mlu',
        marks=pytest.mark.skipif(
            not IS_MLU_AVAILABLE, reason='requires MLU support'))
])
def test_ball_query(device):
    new_xyz = torch.tensor(
        [[[-0.0740, 1.3147, -1.3625], [-2.2769, 2.7817, -0.2334],
          [-0.4003, 2.4666, -0.5116], [-0.0740, 1.3147, -1.3625],
          [-0.0740, 1.3147, -1.3625]],
         [[-2.0289, 2.4952, -0.1708], [-2.0668, 6.0278, -0.4875],
          [0.4066, 1.4211, -0.2947], [-2.0289, 2.4952, -0.1708],
          [-2.0289, 2.4952, -0.1708]]],
        device=device)

    xyz = torch.tensor(
        [[[-0.0740, 1.3147, -1.3625], [0.5555, 1.0399, -1.3634],
          [-0.4003, 2.4666, -0.5116], [-0.5251, 2.4379, -0.8466],
          [-0.9691, 1.1418, -1.3733], [-0.2232, 0.9561, -1.3626],
          [-2.2769, 2.7817, -0.2334], [-0.2822, 1.3192, -1.3645],
          [0.1533, 1.5024, -1.0432], [0.4917, 1.1529, -1.3496]],
         [[-2.0289, 2.4952, -0.1708], [-0.7188, 0.9956, -0.5096],
          [-2.0668, 6.0278, -0.4875], [-1.9304, 3.3092, 0.6610],
          [0.0949, 1.4332, 0.3140], [-1.2879, 2.0008, -0.7791],
          [-0.7252, 0.9611, -0.6371], [0.4066, 1.4211, -0.2947],
          [0.3220, 1.4447, 0.3548], [-0.9744, 2.3856, -1.2000]]],
        device=device)

    idx = ball_query(0, 0.2, 5, xyz, new_xyz)
    expected_idx = torch.tensor(
        [[[0, 0, 0, 0, 0], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        device=device)
    assert torch.all(idx == expected_idx)

    # test dilated ball query
    idx = ball_query(0.2, 0.4, 5, xyz, new_xyz)
    expected_idx = torch.tensor(
        [[[0, 5, 7, 0, 0], [6, 6, 6, 6, 6], [2, 3, 2, 2, 2], [0, 5, 7, 0, 0],
          [0, 5, 7, 0, 0]],
         [[0, 0, 0, 0, 0], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]],
        device=device)
    assert torch.all(idx == expected_idx)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_stack_ball_query():
    new_xyz = torch.tensor([[-0.0740, 1.3147, -1.3625],
                            [-2.2769, 2.7817, -0.2334],
                            [-0.4003, 2.4666, -0.5116],
                            [-0.0740, 1.3147, -1.3625],
                            [-0.0740, 1.3147, -1.3625],
                            [-2.0289, 2.4952, -0.1708],
                            [-2.0668, 6.0278, -0.4875],
                            [0.4066, 1.4211, -0.2947],
                            [-2.0289, 2.4952, -0.1708],
                            [-2.0289, 2.4952, -0.1708]]).cuda()
    new_xyz_batch_cnt = torch.tensor([5, 5], dtype=torch.int32).cuda()
    xyz = torch.tensor([[-0.0740, 1.3147, -1.3625], [0.5555, 1.0399, -1.3634],
                        [-0.4003, 2.4666, -0.5116], [-0.5251, 2.4379, -0.8466],
                        [-0.9691, 1.1418, -1.3733], [-0.2232, 0.9561, -1.3626],
                        [-2.2769, 2.7817, -0.2334], [-0.2822, 1.3192, -1.3645],
                        [0.1533, 1.5024, -1.0432], [0.4917, 1.1529, -1.3496],
                        [-2.0289, 2.4952, -0.1708], [-0.7188, 0.9956, -0.5096],
                        [-2.0668, 6.0278, -0.4875], [-1.9304, 3.3092, 0.6610],
                        [0.0949, 1.4332, 0.3140], [-1.2879, 2.0008, -0.7791],
                        [-0.7252, 0.9611, -0.6371], [0.4066, 1.4211, -0.2947],
                        [0.3220, 1.4447, 0.3548], [-0.9744, 2.3856,
                                                   -1.2000]]).cuda()
    xyz_batch_cnt = torch.tensor([10, 10], dtype=torch.int32).cuda()
    idx = ball_query(0, 0.2, 5, xyz, new_xyz, xyz_batch_cnt, new_xyz_batch_cnt)
    expected_idx = torch.tensor([[0, 0, 0, 0, 0], [6, 6, 6, 6, 6],
                                 [2, 2, 2, 2, 2], [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                                 [2, 2, 2, 2, 2], [7, 7, 7, 7, 7],
                                 [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]).cuda()
    assert torch.all(idx == expected_idx)

    xyz = xyz.double()
    new_xyz = new_xyz.double()
    expected_idx = expected_idx.double()
    idx = ball_query(0, 0.2, 5, xyz, new_xyz, xyz_batch_cnt, new_xyz_batch_cnt)
    assert torch.all(idx == expected_idx)

    xyz = xyz.half()
    new_xyz = new_xyz.half()
    expected_idx = expected_idx.half()
    idx = ball_query(0, 0.2, 5, xyz, new_xyz, xyz_batch_cnt, new_xyz_batch_cnt)
    assert torch.all(idx == expected_idx)
