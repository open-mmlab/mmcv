# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MUSA_AVAILABLE

inputs = ([[[
    [1., 2., 5., 6.],
    [3., 4., 7., 8.],
    [9., 10., 13., 14.],
    [11., 12., 15., 16.],
]]], [[0., 0., 0., 1, 0., 2., 0., 3., 0., 3., 3., 2., 3., 1., 3., 0., 3.]])
outputs = ([[[[1., 1.75, 3.5, 5.25], [2.5, 3.25, 5., 6.75],
              [6., 6.75, 8.5, 10.25],
              [9.5, 10.25, 12., 13.75]]]], [[[[1.5625, 1.5625, 1.5625, 0.3125],
                                              [1.5625, 1.5625, 1.5625, 0.3125],
                                              [1.5625, 1.5625, 1.5625, 0.3125],
                                              [0.3125, 0.3125, 0.3125,
                                               0.0625]]]])


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'musa',
        marks=pytest.mark.skipif(
            not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
])
@pytest.mark.parametrize('dtype', [torch.float, torch.double, torch.half])
def test_bezieralign(device, dtype):
    try:
        from mmcv.ops import bezier_align
    except ModuleNotFoundError:
        pytest.skip('test requires compilation')
    pool_h = 4
    pool_w = 4
    spatial_scale = 1.0
    sampling_ratio = 1
    np_input = np.array(inputs[0])
    np_rois = np.array(inputs[1])
    np_output = np.array(outputs[0])
    np_grad = np.array(outputs[1])

    x = torch.tensor(np_input, dtype=dtype, device=device, requires_grad=True)
    rois = torch.tensor(np_rois, dtype=dtype, device=device)

    output = bezier_align(x, rois, (pool_h, pool_w), spatial_scale,
                          sampling_ratio, False)
    output.backward(torch.ones_like(output))
    assert np.allclose(
        output.data.type(torch.float).cpu().numpy(), np_output, atol=1e-3)
    assert np.allclose(
        x.grad.data.type(torch.float).cpu().numpy(), np_grad, atol=1e-3)
