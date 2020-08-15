import os

import numpy as np
import pytest
import torch

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck

    _USING_PARROTS = False

cur_dir = os.path.dirname(os.path.abspath(__file__))

inputs = ([[[[0.4369, -3.7571], [-1.1835, -1.6374], [0.9534, -0.1321]],
            [[-0.4658, 0.2162], [-0.8135, -0.3903], [-0.1720, -0.0599]],
            [[0.4851, 1.8224], [0.8973, 0.3779], [2.3454, 1.0319]],
            [[0.0420, 0.3574], [0.7641, 0.2384], [0.2759, 0.4931]]],
           [[[-0.5897, 0.7544], [1.0593, 0.8388], [-0.5732, 0.5692]],
            [[-0.6766, -1.4657], [1.2362, 0.4913], [-1.1820, -1.4341]],
            [[0.6476, -0.7391], [1.4314, -0.3522], [0.8401, -0.7757]],
            [[1.4306, 0.9726], [1.0518, -0.8820], [-0.5129, -0.7876]]]])

shifts = [([[1, 0, 1, -2], [-2, 1, -1, 1]]), ([[2, 1, 2, -1], [-1, 2, 0, 2]])]

outputs = [([[[[0.4369, -3.7571], [-1.1835, -1.6374], [0.9534, -0.1321]],
              [[-0.4658, 0.2162], [-0.8135, -0.3903], [-0.1720, -0.0599]],
              [[0.4851, 1.8224], [0.8973, 0.3779], [2.3454, 1.0319]],
              [[0.0420, 0.3574], [0.7641, 0.2384], [0.2759, 0.4931]]],
             [[[0.6476, -0.7391], [1.4314, -0.3522], [0.8401, -0.7757]],
              [[1.4306, 0.9726], [1.0518, -0.8820], [-0.5129, -0.7876]],
              [[0.0000, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000]],
              [[0.0000, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000]]]]),
           ([[[[0.4369, -3.7571], [-1.1835, -1.6374], [0.9534, -0.1321]],
              [[-0.4658, 0.2162], [-0.8135, -0.3903], [-0.1720, -0.0599]],
              [[0.4851, 1.8224], [0.8973, 0.3779], [2.3454, 1.0319]],
              [[0.0420, 0.3574], [0.7641, 0.2384], [0.2759, 0.4931]]],
             [[[-0.6766, -1.4657], [1.2362, 0.4913], [-1.1820, -1.4341]],
              [[0.6476, -0.7391], [1.4314, -0.3522], [0.8401, -0.7757]],
              [[1.4306, 0.9726], [1.0518, -0.8820], [-0.5129, -0.7876]],
              [[0.0000, 0.0000], [0.0000, 0.0000], [0.0000, 0.0000]]]])]

grads = [[[[[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]],
           [[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]]],
          [[[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]],
           [[0., 0.], [0., 0.], [0., 0.]], [[0., 0.], [0., 0.], [0., 0.]]]],
         [[[[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]],
           [[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]]],
          [[[1., 1.], [1., 1.], [1., 1.]], [[1., 1.], [1., 1.], [1., 1.]],
           [[1., 1.], [1., 1.], [1., 1.]], [[0., 0.], [0., 0.], [0., 0.]]]]]


def _test_tinshift_gradcheck(dtype):
    try:
        from mmcv.ops import tin_shift
    except ModuleNotFoundError:
        pytest.skip('TinShift op is not successfully compiled')

    if dtype == torch.half:
        pytest.skip('"add_cpu/sub_cpu" not implemented for Half')

    for shift in shifts:
        np_input = np.array(inputs)
        np_shift = np.array(shift)

        x = torch.tensor(
            np_input, dtype=dtype, device='cuda', requires_grad=True)
        shift = torch.tensor(np_shift, device='cuda').int()
        if torch.__version__ == 'parrots':
            gradcheck(tin_shift, (x, shift))
        else:
            gradcheck(tin_shift, (x, shift), atol=1, rtol=0.1)


def _test_tinshift_allclose(dtype):
    try:
        from mmcv.ops import tin_shift
    except ModuleNotFoundError:
        pytest.skip('TinShift op is not successfully compiled')

    for shift, output, grad in zip(shifts, outputs, grads):
        np_input = np.array(inputs)
        np_shift = np.array(shift)
        np_output = np.array(output)
        np_grad = np.array(grad)

        x = torch.tensor(
            np_input, dtype=dtype, device='cuda', requires_grad=True)
        shift = torch.tensor(np_shift, device='cuda').int()

        output = tin_shift(x, shift)
        output.backward(torch.ones_like(output))
        assert np.allclose(
            output.data.type(torch.float).cpu().numpy(), np_output, 1e-3)
        assert np.allclose(
            x.grad.data.type(torch.float).cpu().numpy(), np_grad, 1e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('dtype', [torch.float, torch.double, torch.half])
def test_tinshift(dtype):
    _test_tinshift_allclose(dtype=dtype)
    _test_tinshift_gradcheck(dtype=dtype)
