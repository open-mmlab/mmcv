# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import three_nn
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE

known = [[[-1.8373, 3.5605, -0.7867], [0.7615, 2.9420, 0.2314],
          [-0.6503, 3.6637, -1.0622], [-1.8373, 3.5605, -0.7867],
          [-1.8373, 3.5605, -0.7867]],
         [[-1.3399, 1.9991, -0.3698], [-0.0799, 0.9698, -0.8457],
          [0.0858, 2.4721, -0.1928], [-1.3399, 1.9991, -0.3698],
          [-1.3399, 1.9991, -0.3698]]]

unknown = [[[-1.8373, 3.5605, -0.7867], [0.7615, 2.9420, 0.2314],
            [-0.6503, 3.6637, -1.0622], [-1.5237, 2.3976, -0.8097],
            [-0.0722, 3.4017, -0.2880], [0.5198, 3.0661, -0.4605],
            [-2.0185, 3.5019, -0.3236], [0.5098, 3.1020, 0.5799],
            [-1.6137, 3.8443, -0.5269], [0.7341, 2.9626, -0.3189]],
           [[-1.3399, 1.9991, -0.3698], [-0.0799, 0.9698, -0.8457],
            [0.0858, 2.4721, -0.1928], [-0.9022, 1.6560, -1.3090],
            [0.1156, 1.6901, -0.4366], [-0.6477, 2.3576, -0.1563],
            [-0.8482, 1.1466, -1.2704], [-0.8753, 2.0845, -0.3460],
            [-0.5621, 1.4233, -1.2858], [-0.5883, 1.3114, -1.2899]]]

expected_dist = [[[0.0000, 0.0000, 0.0000], [0.0000, 2.0463, 2.8588],
                  [0.0000, 1.2229, 1.2229], [1.2047, 1.2047, 1.2047],
                  [1.0011, 1.0845, 1.8411], [0.7433, 1.4451, 2.4304],
                  [0.5007, 0.5007, 0.5007], [0.4587, 2.0875, 2.7544],
                  [0.4450, 0.4450, 0.4450], [0.5514, 1.7206, 2.6811]],
                 [[0.0000, 0.0000, 0.0000], [0.0000, 1.6464, 1.6952],
                  [0.0000, 1.5125, 1.5125], [1.0915, 1.0915, 1.0915],
                  [0.8197, 0.8511, 1.4894], [0.7433, 0.8082, 0.8082],
                  [0.8955, 1.3340, 1.3340], [0.4730, 0.4730, 0.4730],
                  [0.7949, 1.3325, 1.3325], [0.7566, 1.3727, 1.3727]]]

expected_idx = [[[0, 3, 4], [1, 2, 0], [2, 0, 3], [0, 3, 4], [2, 1, 0],
                 [1, 2, 0], [0, 3, 4], [1, 2, 0], [0, 3, 4], [1, 2, 0]],
                [[0, 3, 4], [1, 2, 0], [2, 0, 3], [0, 3, 4], [2, 1, 0],
                 [2, 0, 3], [1, 0, 3], [0, 3, 4], [1, 0, 3], [1, 0, 3]]]


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
@pytest.mark.parametrize('dtype,rtol', [(torch.float, 1e-8),
                                        (torch.half, 1e-3)])
def test_three_nn(device, dtype, rtol):
    dtype = torch.float
    known_t = torch.tensor(known, dtype=dtype, device=device)
    unknown_t = torch.tensor(unknown, dtype=dtype, device=device)

    dist_t, idx_t = three_nn(unknown_t, known_t)
    expected_dist_t = torch.tensor(expected_dist, dtype=dtype, device=device)
    expected_idx_t = torch.tensor(expected_idx, device=device)

    assert torch.allclose(dist_t, expected_dist_t, atol=1e-4, rtol=rtol)
    assert torch.all(idx_t == expected_idx_t)
