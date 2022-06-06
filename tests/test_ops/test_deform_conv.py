# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
import torch

from mmcv.utils import TORCH_VERSION, digit_version

try:
    # If PyTorch version >= 1.6.0 and fp16 is enabled, torch.cuda.amp.autocast
    # would be imported and used; we should test if our modules support it.
    from torch.cuda.amp import autocast
except ImportError:
    pass

input = [[[[1., 2., 3.], [0., 1., 2.], [3., 5., 2.]]]]
offset_weight = [[[0.1, 0.4, 0.6, 0.1]], [[0.3, 0.2, 0.1, 0.3]],
                 [[0.5, 0.5, 0.2, 0.8]], [[0.8, 0.3, 0.9, 0.1]],
                 [[0.3, 0.1, 0.2, 0.5]], [[0.3, 0.7, 0.5, 0.3]],
                 [[0.6, 0.2, 0.5, 0.3]], [[0.4, 0.1, 0.8, 0.4]]]
offset_bias = [0.7, 0.1, 0.8, 0.5, 0.6, 0.5, 0.4, 0.7]
deform_weight = [[[0.4, 0.2, 0.1, 0.9]]]

gt_out = [[[[1.650, 0.], [0.000, 0.]]]]
gt_x_grad = [[[[-0.666, 0.204, 0.000], [0.030, -0.416, 0.012],
               [0.000, 0.252, 0.129]]]]
gt_offset_weight_grad = [[[[1.44, 2.88], [0.00, 1.44]]],
                         [[[-0.72, -1.44], [0.00, -0.72]]],
                         [[[0.00, 0.00], [0.00, 0.00]]],
                         [[[0.00, 0.00], [0.00, 0.00]]],
                         [[[-0.10, -0.20], [0.00, -0.10]]],
                         [[[-0.08, -0.16], [0.00, -0.08]]],
                         [[[-0.54, -1.08], [0.00, -0.54]]],
                         [[[-0.54, -1.08], [0.00, -0.54]]]]
gt_offset_bias_grad = [1.44, -0.72, 0., 0., -0.10, -0.08, -0.54, -0.54],
gt_deform_weight_grad = [[[[3.62, 0.], [0.40, 0.18]]]]


class TestDeformconv:

    def _test_deformconv(self,
                         dtype=torch.float,
                         threshold=1e-3,
                         device='cuda',
                         batch_size=10,
                         im2col_step=2):
        if not torch.cuda.is_available() and device == 'cuda':
            pytest.skip('test requires GPU')
        from mmcv.ops import DeformConv2dPack
        c_in = 1
        c_out = 1
        batch_size = 10
        repeated_input = np.repeat(input, batch_size, axis=0)
        repeated_gt_out = np.repeat(gt_out, batch_size, axis=0)
        repeated_gt_x_grad = np.repeat(gt_x_grad, batch_size, axis=0)
        x = torch.tensor(repeated_input, device=device, dtype=dtype)
        x.requires_grad = True
        model = DeformConv2dPack(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=2,
            stride=1,
            padding=0,
            im2col_step=im2col_step)
        model.conv_offset.weight.data = torch.nn.Parameter(
            torch.Tensor(offset_weight).reshape(8, 1, 2, 2))
        model.conv_offset.bias.data = torch.nn.Parameter(
            torch.Tensor(offset_bias).reshape(8))
        model.weight.data = torch.nn.Parameter(
            torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        if device == 'cuda':
            model.cuda()
        model.type(dtype)

        out = model(x)
        out.backward(torch.ones_like(out))

        assert np.allclose(out.data.detach().cpu().numpy(), repeated_gt_out,
                           threshold)
        assert np.allclose(x.grad.detach().cpu().numpy(), repeated_gt_x_grad,
                           threshold)
        # the batch size of the input is increased which results in
        # a larger gradient so we need to divide by the batch_size
        assert np.allclose(
            model.conv_offset.weight.grad.detach().cpu().numpy() / batch_size,
            gt_offset_weight_grad, threshold)
        assert np.allclose(
            model.conv_offset.bias.grad.detach().cpu().numpy() / batch_size,
            gt_offset_bias_grad, threshold)
        assert np.allclose(
            model.weight.grad.detach().cpu().numpy() / batch_size,
            gt_deform_weight_grad, threshold)

        from mmcv.ops import DeformConv2d

        # test bias
        model = DeformConv2d(1, 1, 2, stride=1, padding=0)
        assert not hasattr(model, 'bias')
        # test bias=True
        with pytest.raises(AssertionError):
            model = DeformConv2d(1, 1, 2, stride=1, padding=0, bias=True)
        # test in_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 2, 3, groups=2)
        # test out_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 4, 3, groups=3)

    def _test_amp_deformconv(self,
                             input_dtype,
                             threshold=1e-3,
                             batch_size=10,
                             im2col_step=2):
        """The function to test amp released on pytorch 1.6.0.

        The type of input data might be torch.float or torch.half,
        so we should test deform_conv in both cases. With amp, the
        data type of model will NOT be set manually.

        Args:
            input_dtype: torch.float or torch.half.
            threshold: the same as above function.
        """
        if not torch.cuda.is_available():
            return
        from mmcv.ops import DeformConv2dPack
        c_in = 1
        c_out = 1
        repeated_input = np.repeat(input, batch_size, axis=0)
        repeated_gt_out = np.repeat(gt_out, batch_size, axis=0)
        repeated_gt_x_grad = np.repeat(gt_x_grad, batch_size, axis=0)
        x = torch.Tensor(repeated_input).cuda().type(input_dtype)
        x.requires_grad = True
        model = DeformConv2dPack(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=2,
            stride=1,
            padding=0,
            im2col_step=im2col_step)
        model.conv_offset.weight.data = torch.nn.Parameter(
            torch.Tensor(offset_weight).reshape(8, 1, 2, 2))
        model.conv_offset.bias.data = torch.nn.Parameter(
            torch.Tensor(offset_bias).reshape(8))
        model.weight.data = torch.nn.Parameter(
            torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        model.cuda()

        out = model(x)
        out.backward(torch.ones_like(out))

        assert np.allclose(out.data.detach().cpu().numpy(), repeated_gt_out,
                           threshold)
        assert np.allclose(x.grad.detach().cpu().numpy(), repeated_gt_x_grad,
                           threshold)
        assert np.allclose(
            model.conv_offset.weight.grad.detach().cpu().numpy() / batch_size,
            gt_offset_weight_grad, threshold)
        assert np.allclose(
            model.conv_offset.bias.grad.detach().cpu().numpy() / batch_size,
            gt_offset_bias_grad, threshold)
        assert np.allclose(
            model.weight.grad.detach().cpu().numpy() / batch_size,
            gt_deform_weight_grad, threshold)

        from mmcv.ops import DeformConv2d

        # test bias
        model = DeformConv2d(1, 1, 2, stride=1, padding=0)
        assert not hasattr(model, 'bias')
        # test bias=True
        with pytest.raises(AssertionError):
            model = DeformConv2d(1, 1, 2, stride=1, padding=0, bias=True)
        # test in_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 2, 3, groups=2)
        # test out_channels % group != 0
        with pytest.raises(AssertionError):
            model = DeformConv2d(3, 4, 3, groups=3)

    def test_deformconv(self):
        self._test_deformconv(torch.double, device='cpu')
        self._test_deformconv(torch.float, device='cpu', threshold=1e-1)
        self._test_deformconv(torch.double)
        self._test_deformconv(torch.float)
        self._test_deformconv(torch.half, threshold=1e-1)
        # test batch_size < im2col_step
        self._test_deformconv(torch.float, batch_size=1, im2col_step=2)
        # test bach_size % im2col_step != 0
        with pytest.raises(
                AssertionError,
                match='batch size must be divisible by im2col_step'):
            self._test_deformconv(torch.float, batch_size=10, im2col_step=3)

        # test amp when torch version >= '1.6.0', the type of
        # input data for deformconv might be torch.float or torch.half
        if (TORCH_VERSION != 'parrots'
                and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
            with autocast(enabled=True):
                self._test_amp_deformconv(torch.float, 1e-1)
                self._test_amp_deformconv(torch.half, 1e-1)
