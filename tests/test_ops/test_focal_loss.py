# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import (IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MUSA_AVAILABLE,
                        IS_NPU_AVAILABLE)

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck
    _USING_PARROTS = False

# torch.set_printoptions(precision=8, threshold=100)

inputs = [
    ([[1., 0], [0, 1.]], [0, 1]),
    ([[1., 0, -1.], [0, 1., 2.]], [2, 1]),
    ([[1e-6, 2e-6, 3e-6], [4e-6, 5e-5, 6e-4], [7e-3, 8e-2, 9e-1]], [1, 2, 0]),
]

softmax_outputs = [(0.00566451, [[-0.00657264, 0.00657264],
                                 [0.00657264, -0.00657264]]),
                   (0.34956908, [[0.10165970, 0.03739851, -0.13905823],
                                 [0.01227554, -0.10298023, 0.09070466]]),
                   (0.15754992, [[0.02590877, -0.05181759, 0.02590882],
                                 [0.02589641, 0.02589760, -0.05179400],
                                 [-0.07307514, 0.02234372, 0.05073142]])]

sigmoid_outputs = [(0.13562961, [[-0.00657264, 0.11185755],
                                 [0.11185755, -0.00657264]]),
                   (1.10251057, [[0.28808805, 0.11185755, -0.09602935],
                                 [0.11185755, -0.00657264, 0.40376765]]),
                   (0.42287254, [[0.07457182, -0.02485716, 0.07457201],
                                 [0.07457211, 0.07457669, -0.02483728],
                                 [-0.02462499, 0.08277918, 0.18050370]])]


class Testfocalloss:

    def _test_softmax(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import softmax_focal_loss
        alpha = 0.25
        gamma = 2.0
        for case, output in zip(inputs, softmax_outputs):
            np_x = np.array(case[0])
            np_y = np.array(case[1])
            np_x_grad = np.array(output[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()

            loss = softmax_focal_loss(x, y, gamma, alpha, None, 'mean')
            loss.backward()

            assert np.allclose(loss.data.cpu().numpy(), output[0], 1e-2)
            assert np.allclose(x.grad.data.cpu(), np_x_grad, 1e-2)

    def _test_sigmoid(self, device, dtype=torch.float):
        from mmcv.ops import sigmoid_focal_loss
        alpha = 0.25
        gamma = 2.0
        for case, output in zip(inputs, sigmoid_outputs):
            np_x = np.array(case[0])
            np_y = np.array(case[1])
            np_x_grad = np.array(output[1])

            x = torch.from_numpy(np_x).to(device).type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).to(device).long()

            loss = sigmoid_focal_loss(x, y, gamma, alpha, None, 'mean')
            loss.backward()

            assert np.allclose(loss.data.cpu().numpy(), output[0], 1e-2)
            assert np.allclose(x.grad.data.cpu(), np_x_grad, 1e-2)

    def _test_grad_softmax(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import SoftmaxFocalLoss
        alpha = 0.25
        gamma = 2.0
        for case in inputs:
            np_x = np.array(case[0])
            np_y = np.array(case[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()

            floss = SoftmaxFocalLoss(gamma, alpha)
            if _USING_PARROTS:
                # gradcheck(floss, (x, y),
                #           no_grads=[y])
                pass
            else:
                gradcheck(floss, (x, y), eps=1e-2, atol=1e-2)

    def _test_grad_sigmoid(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import SigmoidFocalLoss
        alpha = 0.25
        gamma = 2.0
        for case in inputs:
            np_x = np.array(case[0])
            np_y = np.array(case[1])

            x = torch.from_numpy(np_x).cuda().type(dtype)
            x.requires_grad_()
            y = torch.from_numpy(np_y).cuda().long()

            floss = SigmoidFocalLoss(gamma, alpha)
            if _USING_PARROTS:
                # gradcheck(floss, (x, y),
                #           no_grads=[y])
                pass
            else:
                gradcheck(floss, (x, y), eps=1e-2, atol=1e-2)

    def test_softmax_float(self):
        self._test_softmax(dtype=torch.float)

    def test_softmax_half(self):
        self._test_softmax(dtype=torch.half)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support')),
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'musa',
            marks=pytest.mark.skipif(
                not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
    ])
    def test_sigmoid_float(self, device):
        self._test_sigmoid(device=device, dtype=torch.float)

    @pytest.mark.parametrize('device', [
        pytest.param(
            'npu',
            marks=pytest.mark.skipif(
                not IS_NPU_AVAILABLE, reason='requires NPU support')),
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
        pytest.param(
            'mlu',
            marks=pytest.mark.skipif(
                not IS_MLU_AVAILABLE, reason='requires MLU support')),
        pytest.param(
            'musa',
            marks=pytest.mark.skipif(
                not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
    ])
    def test_sigmoid_half(self, device):
        self._test_sigmoid(device, dtype=torch.half)

    def test_grad_softmax_float(self):
        self._test_grad_softmax(dtype=torch.float)

    def test_grad_sigmoid_float(self):
        self._test_grad_sigmoid(dtype=torch.float)

    @pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
    @pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double])
    @pytest.mark.parametrize('gamma', [0., 2., 2.5])
    @pytest.mark.parametrize('alpha', [0., 0.25, 0.995, 1.])
    @pytest.mark.parametrize('weighted', [False, True])
    @pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
    def test_sigmoid_large_logits(self, dtype, gamma, alpha, weighted,
                                  reduction):
        from mmcv.ops import sigmoid_focal_loss

        # Cover target and non-target logits near half-precision sigmoid
        # saturation, and values beyond float/double saturation.
        x = torch.tensor(
            [[10., -10., 8.], [-8., -20., 20.], [100., -100., -100.],
             [-1000., 1000., 0.], [0.1, -0.2, 2.5], [-20., 10., 1000.]],
            device='cuda',
            dtype=dtype,
            requires_grad=True)
        y = torch.tensor([0, 1, 2, 0, 1, 2], device='cuda')
        weight = x.new_tensor([0.5, 1., 1.5]) if weighted else None
        loss = sigmoid_focal_loss(x, y, gamma, alpha, weight, reduction)
        loss.sum().backward()

        # Use an independent, stable BCE-with-logits reference in double.
        reference_x = x.detach().double().requires_grad_()
        target = torch.nn.functional.one_hot(y, x.size(1)).double()
        probability = reference_x.sigmoid()
        modulation = (1 - probability) * target + probability * (1 - target)
        reference = torch.nn.functional.binary_cross_entropy_with_logits(
            reference_x, target, reduction='none')
        reference = reference * modulation.pow(gamma) * (
            alpha * target + (1 - alpha) * (1 - target))
        if weight is not None:
            reference = reference * weight.double()[y, None]
        if reduction == 'mean':
            reference = reference.sum() / x.size(0)
        elif reduction == 'sum':
            reference = reference.sum()
        reference.sum().backward()

        assert loss.dtype == dtype
        assert torch.isfinite(loss).all()
        assert torch.isfinite(x.grad).all()
        rtol, atol = (2e-3, 2e-5) if dtype == torch.half else (1e-5, 1e-7)
        assert torch.allclose(loss.double(), reference, rtol=rtol, atol=atol)
        assert torch.allclose(
            x.grad.double(), reference_x.grad, rtol=rtol, atol=atol)

    @pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
    @pytest.mark.parametrize('dtype', [torch.half, torch.float, torch.double])
    def test_sigmoid_large_logits_background(self, dtype):
        from mmcv.ops import sigmoid_focal_loss

        x = torch.tensor([[100., -100.]],
                         device='cuda',
                         dtype=dtype,
                         requires_grad=True)
        # num_classes is the background label: every channel is negative.
        target = torch.tensor([x.size(1)], device='cuda')
        loss = sigmoid_focal_loss(x, target, 2., 0.25, None, 'none')
        loss.sum().backward()
        assert torch.allclose(loss, x.new_tensor([[75., 0.]]))
        assert torch.allclose(x.grad, x.new_tensor([[0.75, 0.]]))
