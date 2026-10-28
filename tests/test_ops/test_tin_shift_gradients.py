# Copyright (c) OpenMMLab. All rights reserved.
"""Independent TinShift mapping and derivative regressions."""
import pytest
import torch

from mmcv.ops import tin_shift
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MUSA_AVAILABLE

DEVICES = [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'musa',
        marks=pytest.mark.skipif(
            not IS_MUSA_AVAILABLE, reason='requires MUSA support')),
]


def reference(x, shift):
    # Definition: y[n, t + shift[n, group], c] = x[n, t, c].
    # CPU PyTorch autograd provides the inverse mapping for the derivative.
    out = x * 0
    length, channels = x.shape[1:3]
    width = channels // shift.shape[1]
    for batch in range(x.shape[0]):
        for group in range(shift.shape[1]):
            offset = int(shift[batch, group])
            if abs(offset) >= length:
                continue
            src = slice(max(-offset, 0), min(length - offset, length))
            dst = slice(max(offset, 0), min(length + offset, length))
            channel = slice(group * width, (group + 1) * width)
            out[batch, dst, channel] = x[batch, src, channel]
    return out


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype',
                         [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize('offsets', [
    [[1, 0, 1, -2], [-2, 1, -1, 1]],
    [[2, 1, 2, -1], [-1, 2, 0, 2]],
    [[0, 0, 0, 0], [0, 0, 0, 0]],
    [[4, -4, 5, -5], [-5, 5, -4, 4]],
])
@pytest.mark.parametrize('layout', ['contiguous', 'strided', 'expanded'])
def test_mapping_and_gradient(device, dtype, offsets, layout):
    if device == 'musa' and dtype == torch.float64:
        pytest.skip('MUSA does not support float64')
    torch.manual_seed(20261028)
    x = torch.randn(2, 4, 8, 2, dtype=dtype)
    if layout == 'strided':
        x = torch.randn(2, 4, 8, 4, dtype=dtype)[..., ::2]
    shift = torch.tensor(offsets, dtype=torch.int32)
    cpu = x.double().detach().requires_grad_()
    if layout == 'strided':
        gpu = torch.empty(2, 4, 8, 4, device=device, dtype=dtype)[..., ::2]
        gpu.copy_(x.to(device))
    else:
        gpu = x.to(device)
    gpu = gpu.detach().requires_grad_()
    expected = reference(cpu, shift)
    actual = tin_shift(gpu, shift.to(device))
    if layout == 'expanded':
        grad = torch.ones((), dtype=dtype).expand_as(x)
    elif layout == 'strided':
        grad = torch.randn(2, 4, 8, 4, dtype=dtype)[..., ::2]
    else:
        grad = torch.randn_like(x)
    expected.backward(grad.double())
    if layout == 'expanded':
        grad_gpu = torch.ones((), device=device, dtype=dtype).expand_as(actual)
    elif layout == 'strided':
        grad_gpu = torch.empty(
            2, 4, 8, 4, device=device, dtype=dtype)[..., ::2]
        grad_gpu.copy_(grad.to(device))
    else:
        grad_gpu = grad.to(device)
    actual.backward(grad_gpu)
    # Shift/copy contains no arithmetic: exact equality is appropriate.
    assert torch.equal(actual.cpu(), expected.detach().to(dtype))
    assert torch.equal(gpu.grad.cpu(), cpu.grad.to(dtype))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype,eps', [(torch.float64, 1e-6),
                                       (torch.float32, 1e-3),
                                       (torch.float16, 0.0078125)])
def test_strict_gradcheck(device, dtype, eps):
    if device == 'musa' and dtype == torch.float64:
        pytest.skip('MUSA does not support float64')
    # Dyadic inputs make Half finite-difference perturbations representable.
    x = (torch.arange(32).reshape(1, 4, 4, 2) / 32).to(device, dtype)
    x.requires_grad_()
    shift = torch.tensor([[1, -1]], device=device, dtype=torch.int32)
    assert torch.autograd.gradcheck(
        tin_shift, (x, shift), eps=eps, atol=1e-3, rtol=1e-2)
