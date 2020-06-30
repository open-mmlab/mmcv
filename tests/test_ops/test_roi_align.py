import numpy as np
import pytest
import torch

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck
    _USING_PARROTS = False

# yapf:disable
inputs = [([[[[1., 2.], [3., 4.]]]],
           [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2.], [3., 4.]],
             [[4., 3.], [2., 1.]]]],
           [[0., 0., 0., 1., 1.]]),
          ([[[[1., 2., 5., 6.], [3., 4., 7., 8.],
              [9., 10., 13., 14.], [11., 12., 15., 16.]]]],
           [[0., 0., 0., 3., 3.]])]
outputs = [([[[[1.0, 1.25], [1.5, 1.75]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.0, 1.25], [1.5, 1.75]],
              [[4.0, 3.75], [3.5, 3.25]]]],
            [[[[3.0625, 0.4375], [0.4375, 0.0625]],
              [[3.0625, 0.4375], [0.4375, 0.0625]]]]),
           ([[[[1.9375, 4.75], [7.5625, 10.375]]]],
            [[[[0.47265625, 0.42968750, 0.42968750, 0.04296875],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.42968750, 0.39062500, 0.39062500, 0.03906250],
               [0.04296875, 0.03906250, 0.03906250, 0.00390625]]]])]
# yapf:enable

pool_h = 2
pool_w = 2
spatial_scale = 1.0
sampling_ratio = 2


def _test_roialign_gradcheck(device, dtype):
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('test requires GPU')
    try:
        from mmcv.ops import RoIAlign
    except ModuleNotFoundError:
        pytest.skip('RoIAlign op is not successfully compiled')
    if dtype is torch.half:
        pytest.skip('grad check does not support fp16')
    for case in inputs:
        np_input = np.array(case[0])
        np_rois = np.array(case[1])

        x = torch.tensor(
            np_input, dtype=dtype, device=device, requires_grad=True)
        rois = torch.tensor(np_rois, dtype=dtype, device=device)

        froipool = RoIAlign((pool_h, pool_w), spatial_scale, sampling_ratio)

        gradcheck(froipool, (x, rois), eps=1e-5, atol=1e-5)


def _test_roialign_allclose(device, dtype):
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('test requires GPU')
    try:
        from mmcv.ops import roi_align
    except ModuleNotFoundError:
        pytest.skip('test requires compilation')
    pool_h = 2
    pool_w = 2
    spatial_scale = 1.0
    sampling_ratio = 2

    for case, output in zip(inputs, outputs):
        np_input = np.array(case[0])
        np_rois = np.array(case[1])
        np_output = np.array(output[0])
        np_grad = np.array(output[1])

        x = torch.tensor(
            np_input, dtype=dtype, device=device, requires_grad=True)
        rois = torch.tensor(np_rois, dtype=dtype, device=device)

        output = roi_align(x, rois, (pool_h, pool_w), spatial_scale,
                           sampling_ratio, 'avg', True)
        output.backward(torch.ones_like(output))
        assert np.allclose(
            output.data.type(torch.float).cpu().numpy(), np_output, atol=1e-3)
        assert np.allclose(
            x.grad.data.type(torch.float).cpu().numpy(), np_grad, atol=1e-3)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double, torch.half])
def test_roialign(device, dtype):
    # check double only
    if dtype is torch.double:
        _test_roialign_gradcheck(device=device, dtype=dtype)
    _test_roialign_allclose(device=device, dtype=dtype)
