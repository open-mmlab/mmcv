# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import knn


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_knn():
    new_xyz = torch.tensor([[[-0.0740, 1.3147, -1.3625],
                             [-2.2769, 2.7817, -0.2334],
                             [-0.4003, 2.4666, -0.5116],
                             [-0.0740, 1.3147, -1.3625],
                             [-0.0740, 1.3147, -1.3625]],
                            [[-2.0289, 2.4952, -0.1708],
                             [-2.0668, 6.0278, -0.4875],
                             [0.4066, 1.4211, -0.2947],
                             [-2.0289, 2.4952, -0.1708],
                             [-2.0289, 2.4952, -0.1708]]]).cuda()

    xyz = torch.tensor([[[-0.0740, 1.3147, -1.3625], [0.5555, 1.0399, -1.3634],
                         [-0.4003, 2.4666,
                          -0.5116], [-0.5251, 2.4379, -0.8466],
                         [-0.9691, 1.1418,
                          -1.3733], [-0.2232, 0.9561, -1.3626],
                         [-2.2769, 2.7817, -0.2334],
                         [-0.2822, 1.3192, -1.3645], [0.1533, 1.5024, -1.0432],
                         [0.4917, 1.1529, -1.3496]],
                        [[-2.0289, 2.4952,
                          -0.1708], [-0.7188, 0.9956, -0.5096],
                         [-2.0668, 6.0278, -0.4875], [-1.9304, 3.3092, 0.6610],
                         [0.0949, 1.4332, 0.3140], [-1.2879, 2.0008, -0.7791],
                         [-0.7252, 0.9611, -0.6371], [0.4066, 1.4211, -0.2947],
                         [0.3220, 1.4447, 0.3548], [-0.9744, 2.3856,
                                                    -1.2000]]]).cuda()

    idx = knn(5, xyz, new_xyz)
    new_xyz_ = new_xyz.unsqueeze(2).repeat(1, 1, xyz.shape[1], 1)
    xyz_ = xyz.unsqueeze(1).repeat(1, new_xyz.shape[1], 1, 1)
    dist = ((new_xyz_ - xyz_) * (new_xyz_ - xyz_)).sum(-1)
    expected_idx = dist.topk(k=5, dim=2, largest=False)[1].transpose(2, 1)
    assert torch.all(idx == expected_idx)

    idx = knn(5,
              xyz.transpose(1, 2).contiguous(),
              new_xyz.transpose(1, 2).contiguous(), True)
    assert torch.all(idx == expected_idx)

    idx = knn(5, xyz, xyz)
    xyz_ = xyz.unsqueeze(2).repeat(1, 1, xyz.shape[1], 1)
    xyz__ = xyz.unsqueeze(1).repeat(1, xyz.shape[1], 1, 1)
    dist = ((xyz_ - xyz__) * (xyz_ - xyz__)).sum(-1)
    expected_idx = dist.topk(k=5, dim=2, largest=False)[1].transpose(2, 1)
    assert torch.all(idx == expected_idx)
