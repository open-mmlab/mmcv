# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np
import pytest
import torch

# [1,4c,h,w]
input_arr = [[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]],
              [[6, 7, 5, 8], [2, 1, 3, 4], [12, 9, 11, 10]],
              [[-2, -3, 2, 0], [-4, -5, 1, -1], [-1, -1, -1, -1]],
              [[0, -1, 2, 1], [-4, -3, -2, -1], [-1, -2, -3, -4]]]]
# [1,h*w,4]
boxes_arr = [[[0, 0, 2, 1], [1, 0, 3, 1], [1, 0, 2, 1], [0, 0, 3, 1],
              [0, 0, 1, 2], [0, 0, 2, 2], [1, 0, 2, 1], [1, 0, 3, 1],
              [0, 1, 1, 2], [0, 0, 3, 2], [1, 0, 3, 2], [2, 0, 3, 2]]]
output_dict = {
    # [1,c,h*w,4] for each value,
    # the output is manually checked for its correctness

    # pool_size=1
    1: [[[[3., 6., 1., 2.], [4., 7., -1., 1.], [3., 7., 1., 2.],
          [4., 6., -1., 1.], [2., 12., -1., -1.], [3., 12., -1., 2.],
          [3., 7., 1., 2.], [4., 7., -1., 1.], [6., 12., -1., -2.],
          [4., 12., -1., 1.], [4., 9., -1., 1.], [4., 11., -1., 1.]]]],

    # pool_size=2
    2: [[[[3., 6., 1., 2.], [4., 7., 1., 1.], [3., 7., 1., 2.],
          [4., 6., -1., 1.], [2., 12., -1., -1.], [3., 12., -1., 2.],
          [3., 7., 1., 2.], [4., 7., 1., 1.], [6., 12., -1., -2.],
          [4., 12., -1., 1.], [4., 9., -1., 1.], [4., 11., -1., 1.]]]],
}
input_grad_dict = {
    # [1,4c,h,w] for each value
    # the grad is manually checked for its correctness

    # pool_size=1
    1: [[[[0., 1., 4., 6.], [0., 1., 0., 0.], [0., 0., 0., 0.]],
         [[2., 4., 0., 0.], [0., 0., 0., 0.], [4., 1., 1., 0.]],
         [[0., 0., 0., 0.], [0., 0., 3., 3.], [0., 2., 1., 3.]],
         [[0., 1., 4., 6.], [0., 0., 0., 0.], [0., 1., 0., 0.]]]],

    # pool_size=2
    2: [[[[0., 1., 4., 6.], [0., 1., 0., 0.], [0., 0., 0., 0.]],
         [[2., 4., 0., 0.], [0., 0., 0., 0.], [4., 1., 1., 0.]],
         [[0., 0., 0., 0.], [0., 0., 5., 1.], [0., 2., 1., 3.]],
         [[0., 1., 4., 6.], [0., 0., 0., 0.], [0., 1., 0., 0.]]]],
}


def _test_border_align_allclose(device, dtype, pool_size):
    if not torch.cuda.is_available() and device == 'cuda':
        pytest.skip('test requires GPU')
    try:
        from mmcv.ops import BorderAlign, border_align
    except ModuleNotFoundError:
        pytest.skip('BorderAlign op is not successfully compiled')

    np_input = np.array(input_arr)
    np_boxes = np.array(boxes_arr)
    np_output = np.array(output_dict[pool_size])
    np_grad = np.array(input_grad_dict[pool_size])

    input = torch.tensor(
        np_input, dtype=dtype, device=device, requires_grad=True)
    boxes = torch.tensor(np_boxes, dtype=dtype, device=device)

    # test for border_align
    input_cp = copy.deepcopy(input)
    output = border_align(input_cp, boxes, pool_size)
    output.backward(torch.ones_like(output))
    assert np.allclose(
        output.data.type(dtype).cpu().numpy(), np_output, atol=1e-5)
    assert np.allclose(
        input_cp.grad.data.type(dtype).cpu().numpy(), np_grad, atol=1e-5)

    # test for BorderAlign
    pool_module = BorderAlign(pool_size)
    output = pool_module(input, boxes)
    output.backward(torch.ones_like(output))
    assert np.allclose(
        output.data.type(dtype).cpu().numpy(), np_output, atol=1e-5)
    assert np.allclose(
        input.grad.data.type(dtype).cpu().numpy(), np_grad, atol=1e-5)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.half, torch.double])
@pytest.mark.parametrize('pool_size', [1, 2])
def test_border_align(device, dtype, pool_size):
    _test_border_align_allclose(device, dtype, pool_size)
