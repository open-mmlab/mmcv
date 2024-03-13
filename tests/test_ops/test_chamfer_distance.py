# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.ops import chamfer_distance
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MUSA_AVAILABLE, IS_NPU_AVAILABLE


def chamfer_distance_forward_groundtruth(xyz1, xyz2, dtype):
    bs, ns, ss = xyz1.shape
    dist1 = np.zeros((bs, ns)).astype(torch_to_np_type(dtype))
    dist2 = np.zeros((bs, ns)).astype(torch_to_np_type(dtype))
    idx1 = np.zeros((bs, ns)).astype('int32')
    idx2 = np.zeros((bs, ns)).astype('int32')
    for b1 in range(bs):
        for n1 in range(ns):
            x1, y1 = xyz1[b1][n1]
            dist1[b1][n1] = 10000000
            for n2 in range(ns):
                x2, y2 = xyz2[b1][n2]
                dst = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                if dist1[b1][n1] > dst:
                    dist1[b1][n1] = dst
                    idx1[b1][n1] = n2
    for b1 in range(bs):
        for n1 in range(ns):
            x1, y1 = xyz2[b1][n1]
            dist2[b1][n1] = 10000000
            for n2 in range(ns):
                x2, y2 = xyz1[b1][n2]
                dst = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
                if dist2[b1][n1] > dst:
                    dist2[b1][n1] = dst
                    idx2[b1][n1] = n2
    return [dist1, dist2, idx1, idx2]


def torch_to_np_type(dtype):
    if dtype == torch.half:
        return np.float16
    elif dtype == torch.float32:
        return np.float32


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'npu',
        marks=pytest.mark.skipif(
            not IS_NPU_AVAILABLE, reason='requires NPU support')),
    pytest.param(
        'musa',
        marks=pytest.mark.skipif(
            not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
])
@pytest.mark.parametrize('dtype', [
    pytest.param(
        torch.half,
        marks=pytest.mark.skipif(
            IS_MUSA_AVAILABLE,
            reason='TODO haowen.han@mthreads.com: not supported yet')),
    torch.float32
])
@pytest.mark.parametrize('shape', [(2, 600, 2), (2, 600, 2)])
def test_chamfer_distance_npu_dynamic_shape(dtype, device, shape):
    if device == 'musa':
        from torch_musa.testing import get_musa_arch
        if get_musa_arch() <= 21:
            return
    bs = shape[0]
    ns = shape[1]
    xyz1 = np.random.uniform(-10.0, 10.0,
                             (bs, ns, 2)).astype(torch_to_np_type(dtype))
    xyz2 = np.random.uniform(-10.0, 10.0,
                             (bs, ns, 2)).astype(torch_to_np_type(dtype))
    xyz1_npu = torch.tensor(xyz1, dtype=dtype).to(device)
    xyz2_npu = torch.tensor(xyz2, dtype=dtype).to(device)
    expected_output = chamfer_distance_forward_groundtruth(xyz1, xyz2, dtype)
    output = chamfer_distance(xyz1_npu, xyz2_npu)
    assert np.allclose(output[0].cpu().numpy(), expected_output[0], 1e-3, 1e-4)
    assert np.allclose(output[1].cpu().numpy(), expected_output[1], 1e-3, 1e-4)
    assert np.allclose(output[2].cpu().numpy(), expected_output[2], 1e-3, 1e-4)
    assert np.allclose(output[3].cpu().numpy(), expected_output[3], 1e-3, 1e-4)
