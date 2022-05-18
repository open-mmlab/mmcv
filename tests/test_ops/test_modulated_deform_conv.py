# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy
import pytest
import torch

from mmcv.utils import TORCH_VERSION, digit_version

try:
    # If PyTorch version >= 1.6.0 and fp16 is enabled, torch.cuda.amp.autocast
    # would be imported and used; we should test if our modules support it.
    from torch.cuda.amp import autocast
except ImportError:
    pass

cur_dir = os.path.dirname(os.path.abspath(__file__))

input_t = [[[[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]]]]
output_t = [[[[0.5, 1.5, 2.5, 1.5], [1.0, 3.0, 5.0, 3.0], [1.0, 3.0, 5.0, 3.0],
              [0.5, 1.5, 2.5, 1.5]]]]
input_grad = [[[[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]]]]
dcn_w_grad = [[[[9., 9.], [9., 9.]]]]
dcn_offset_w_grad = [[[[-7.0, -4.0], [0.0, 0.0]]], [[[-9.0, 7.5], [-6.0,
                                                                   5.0]]],
                     [[[-4.0, -7.0], [0.0, 0.0]]],
                     [[[-7.5, -9.0], [-5.0, -6.0]]],
                     [[[-7.0, -4.0], [-7.0, -4.0]]],
                     [[[-6.0, 5.0], [-9.0, 7.5]]],
                     [[[-4.0, -7.0], [-4.0, -7.0]]],
                     [[[-5.0, -6.0], [-7.5, -9.0]]], [[[10.5, 6.0], [7.0,
                                                                     4.0]]],
                     [[[6.0, 10.5], [4.0, 7.0]]], [[[7.0, 4.0], [10.5, 6.0]]],
                     [[[4.0, 7.0], [6.0, 10.5]]]]
dcn_offset_b_grad = [
    -3.0, -1.5, -3.0, -1.5, -3.0, -1.5, -3.0, -1.5, 4.5, 4.5, 4.5, 4.5
]


class TestMdconv:

    def _test_mdconv(self, dtype=torch.float, device='cuda'):
        if not torch.cuda.is_available() and device == 'cuda':
            pytest.skip('test requires GPU')
        from mmcv.ops import ModulatedDeformConv2dPack
        input = torch.tensor(input_t, dtype=dtype, device=device)
        input.requires_grad = True

        dcn = ModulatedDeformConv2dPack(
            1,
            1,
            kernel_size=(2, 2),
            stride=1,
            padding=1,
            deform_groups=1,
            bias=False)

        if device == 'cuda':
            dcn.cuda()

        dcn.weight.data.fill_(1.)
        dcn.type(dtype)
        output = dcn(input)
        output.sum().backward()
        assert numpy.allclose(output.cpu().detach().numpy(), output_t, 1e-2)
        assert numpy.allclose(input.grad.cpu().detach().numpy(), input_grad,
                              1e-2)
        assert numpy.allclose(dcn.weight.grad.cpu().detach().numpy(),
                              dcn_w_grad, 1e-2)
        assert numpy.allclose(
            dcn.conv_offset.weight.grad.cpu().detach().numpy(),
            dcn_offset_w_grad, 1e-2)
        assert numpy.allclose(dcn.conv_offset.bias.grad.cpu().detach().numpy(),
                              dcn_offset_b_grad, 1e-2)

    def _test_amp_mdconv(self, input_dtype=torch.float):
        """The function to test amp released on pytorch 1.6.0.

        The type of input data might be torch.float or torch.half,
        so we should test mdconv in both cases. With amp, the data
        type of model will NOT be set manually.

        Args:
            input_dtype: torch.float or torch.half.
        """
        if not torch.cuda.is_available():
            return
        from mmcv.ops import ModulatedDeformConv2dPack
        input = torch.tensor(input_t).cuda().type(input_dtype)
        input.requires_grad = True

        dcn = ModulatedDeformConv2dPack(
            1,
            1,
            kernel_size=(2, 2),
            stride=1,
            padding=1,
            deform_groups=1,
            bias=False).cuda()
        dcn.weight.data.fill_(1.)
        output = dcn(input)
        output.sum().backward()
        assert numpy.allclose(output.cpu().detach().numpy(), output_t, 1e-2)
        assert numpy.allclose(input.grad.cpu().detach().numpy(), input_grad,
                              1e-2)
        assert numpy.allclose(dcn.weight.grad.cpu().detach().numpy(),
                              dcn_w_grad, 1e-2)
        assert numpy.allclose(
            dcn.conv_offset.weight.grad.cpu().detach().numpy(),
            dcn_offset_w_grad, 1e-2)
        assert numpy.allclose(dcn.conv_offset.bias.grad.cpu().detach().numpy(),
                              dcn_offset_b_grad, 1e-2)

    def test_mdconv(self):
        self._test_mdconv(torch.double, device='cpu')
        self._test_mdconv(torch.float, device='cpu')
        self._test_mdconv(torch.double)
        self._test_mdconv(torch.float)
        self._test_mdconv(torch.half)

        # test amp when torch version >= '1.6.0', the type of
        # input data for mdconv might be torch.float or torch.half
        if (TORCH_VERSION != 'parrots'
                and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):
            with autocast(enabled=True):
                self._test_amp_mdconv(torch.float)
                self._test_amp_mdconv(torch.half)
