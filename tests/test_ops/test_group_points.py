# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import grouping_operation
from mmcv.utils import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE


@pytest.mark.parametrize('device', [
    pytest.param('cuda',
                 marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE,
                                          reason='requires CUDA support')),
    pytest.param('npu',
                 marks=pytest.mark.skipif(not IS_NPU_AVAILABLE,
                                          reason='requires NPU support'))
])
@pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double])
def test_grouping_points(dtype, device):
    idx = torch.tensor([[[0, 0, 0], [3, 3, 3], [8, 8, 8], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0]],
                        [[0, 0, 0], [6, 6, 6], [9, 9, 9], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0]]]).int().to(device)
    features = torch.tensor([[[
        0.5798, -0.7981, -0.9280, -1.3311, 1.3687, 0.9277, -0.4164, -1.8274,
        0.9268, 0.8414
    ],
                              [
                                  5.4247, 1.5113, 2.3944, 1.4740, 5.0300,
                                  5.1030, 1.9360, 2.1939, 2.1581, 3.4666
                              ],
                              [
                                  -1.6266, -1.0281, -1.0393, -1.6931, -1.3982,
                                  -0.5732, -1.0830, -1.7561, -1.6786, -1.6967
                              ]],
                             [[
                                 -0.0380, -0.1880, -1.5724, 0.6905, -0.3190,
                                 0.7798, -0.3693, -0.9457, -0.2942, -1.8527
                             ],
                              [
                                  1.1773, 1.5009, 2.6399, 5.9242, 1.0962,
                                  2.7346, 6.0865, 1.5555, 4.3303, 2.8229
                              ],
                              [
                                  -0.6646, -0.6870, -0.1125, -0.2224, -0.3445,
                                  -1.4049, 0.4990, -0.7037, -0.9924, 0.0386
                              ]]],
                            dtype=dtype).to(device)

    output = grouping_operation(features, idx)
    expected_output = torch.tensor(
        [[[[0.5798, 0.5798, 0.5798], [-1.3311, -1.3311, -1.3311],
           [0.9268, 0.9268, 0.9268], [0.5798, 0.5798, 0.5798],
           [0.5798, 0.5798, 0.5798], [0.5798, 0.5798, 0.5798]],
          [[5.4247, 5.4247, 5.4247], [1.4740, 1.4740, 1.4740],
           [2.1581, 2.1581, 2.1581], [5.4247, 5.4247, 5.4247],
           [5.4247, 5.4247, 5.4247], [5.4247, 5.4247, 5.4247]],
          [[-1.6266, -1.6266, -1.6266], [-1.6931, -1.6931, -1.6931],
           [-1.6786, -1.6786, -1.6786], [-1.6266, -1.6266, -1.6266],
           [-1.6266, -1.6266, -1.6266], [-1.6266, -1.6266, -1.6266]]],
         [[[-0.0380, -0.0380, -0.0380], [-0.3693, -0.3693, -0.3693],
           [-1.8527, -1.8527, -1.8527], [-0.0380, -0.0380, -0.0380],
           [-0.0380, -0.0380, -0.0380], [-0.0380, -0.0380, -0.0380]],
          [[1.1773, 1.1773, 1.1773], [6.0865, 6.0865, 6.0865],
           [2.8229, 2.8229, 2.8229], [1.1773, 1.1773, 1.1773],
           [1.1773, 1.1773, 1.1773], [1.1773, 1.1773, 1.1773]],
          [[-0.6646, -0.6646, -0.6646], [0.4990, 0.4990, 0.4990],
           [0.0386, 0.0386, 0.0386], [-0.6646, -0.6646, -0.6646],
           [-0.6646, -0.6646, -0.6646], [-0.6646, -0.6646, -0.6646]]]],
        dtype=dtype).to(device)
    assert torch.allclose(output, expected_output)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='requires CUDA support')
@pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double])
def test_stack_grouping_points(dtype):
    idx = torch.tensor([[0, 0, 0], [3, 3, 3], [8, 8, 8], [1, 1, 1], [0, 0, 0],
                        [2, 2, 2], [0, 0, 0], [6, 6, 6], [9, 9, 9], [0, 0, 0],
                        [1, 1, 1], [0, 0, 0]]).int().cuda()
    features = torch.tensor([[
        0.5798, -0.7981, -0.9280, -1.3311, 1.3687, 0.9277, -0.4164, -1.8274,
        0.9268, 0.8414
    ],
                             [
                                 5.4247, 1.5113, 2.3944, 1.4740, 5.0300,
                                 5.1030, 1.9360, 2.1939, 2.1581, 3.4666
                             ],
                             [
                                 -1.6266, -1.0281, -1.0393, -1.6931, -1.3982,
                                 -0.5732, -1.0830, -1.7561, -1.6786, -1.6967
                             ],
                             [
                                 -0.0380, -0.1880, -1.5724, 0.6905, -0.3190,
                                 0.7798, -0.3693, -0.9457, -0.2942, -1.8527
                             ],
                             [
                                 1.1773, 1.5009, 2.6399, 5.9242, 1.0962,
                                 2.7346, 6.0865, 1.5555, 4.3303, 2.8229
                             ],
                             [
                                 -0.6646, -0.6870, -0.1125, -0.2224, -0.3445,
                                 -1.4049, 0.4990, -0.7037, -0.9924, 0.0386
                             ]],
                            dtype=dtype).cuda()
    features_batch_cnt = torch.tensor([3, 3]).int().cuda()
    indices_batch_cnt = torch.tensor([6, 6]).int().cuda()
    output = grouping_operation(features, idx, features_batch_cnt,
                                indices_batch_cnt)
    expected_output = torch.tensor(
        [[[0.5798, 0.5798, 0.5798], [-0.7981, -0.7981, -0.7981],
          [-0.9280, -0.9280, -0.9280], [-1.3311, -1.3311, -1.3311],
          [1.3687, 1.3687, 1.3687], [0.9277, 0.9277, 0.9277],
          [-0.4164, -0.4164, -0.4164], [-1.8274, -1.8274, -1.8274],
          [0.9268, 0.9268, 0.9268], [0.8414, 0.8414, 0.8414]],
         [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
         [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
         [[5.4247, 5.4247, 5.4247], [1.5113, 1.5113, 1.5113],
          [2.3944, 2.3944, 2.3944], [1.4740, 1.4740, 1.4740],
          [5.0300, 5.0300, 5.0300], [5.1030, 5.1030, 5.1030],
          [1.9360, 1.9360, 1.9360], [2.1939, 2.1939, 2.1939],
          [2.1581, 2.1581, 2.1581], [3.4666, 3.4666, 3.4666]],
         [[0.5798, 0.5798, 0.5798], [-0.7981, -0.7981, -0.7981],
          [-0.9280, -0.9280, -0.9280], [-1.3311, -1.3311, -1.3311],
          [1.3687, 1.3687, 1.3687], [0.9277, 0.9277, 0.9277],
          [-0.4164, -0.4164, -0.4164], [-1.8274, -1.8274, -1.8274],
          [0.9268, 0.9268, 0.9268], [0.8414, 0.8414, 0.8414]],
         [[-1.6266, -1.6266, -1.6266], [-1.0281, -1.0281, -1.0281],
          [-1.0393, -1.0393, -1.0393], [-1.6931, -1.6931, -1.6931],
          [-1.3982, -1.3982, -1.3982], [-0.5732, -0.5732, -0.5732],
          [-1.0830, -1.0830, -1.0830], [-1.7561, -1.7561, -1.7561],
          [-1.6786, -1.6786, -1.6786], [-1.6967, -1.6967, -1.6967]],
         [[-0.0380, -0.0380, -0.0380], [-0.1880, -0.1880, -0.1880],
          [-1.5724, -1.5724, -1.5724], [0.6905, 0.6905, 0.6905],
          [-0.3190, -0.3190, -0.3190], [0.7798, 0.7798, 0.7798],
          [-0.3693, -0.3693, -0.3693], [-0.9457, -0.9457, -0.9457],
          [-0.2942, -0.2942, -0.2942], [-1.8527, -1.8527, -1.8527]],
         [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
         [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.0000]],
         [[-0.0380, -0.0380, -0.0380], [-0.1880, -0.1880, -0.1880],
          [-1.5724, -1.5724, -1.5724], [0.6905, 0.6905, 0.6905],
          [-0.3190, -0.3190, -0.3190], [0.7798, 0.7798, 0.7798],
          [-0.3693, -0.3693, -0.3693], [-0.9457, -0.9457, -0.9457],
          [-0.2942, -0.2942, -0.2942], [-1.8527, -1.8527, -1.8527]],
         [[1.1773, 1.1773, 1.1773], [1.5009, 1.5009, 1.5009],
          [2.6399, 2.6399, 2.6399], [5.9242, 5.9242, 5.9242],
          [1.0962, 1.0962, 1.0962], [2.7346, 2.7346, 2.7346],
          [6.0865, 6.0865, 6.0865], [1.5555, 1.5555, 1.5555],
          [4.3303, 4.3303, 4.3303], [2.8229, 2.8229, 2.8229]],
         [[-0.0380, -0.0380, -0.0380], [-0.1880, -0.1880, -0.1880],
          [-1.5724, -1.5724, -1.5724], [0.6905, 0.6905, 0.6905],
          [-0.3190, -0.3190, -0.3190], [0.7798, 0.7798, 0.7798],
          [-0.3693, -0.3693, -0.3693], [-0.9457, -0.9457, -0.9457],
          [-0.2942, -0.2942, -0.2942], [-1.8527, -1.8527, -1.8527]]],
        dtype=dtype).cuda()
    assert torch.allclose(output, expected_output)
