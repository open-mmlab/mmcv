import os

import numpy
import torch

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


class TestMdconv(object):

    def _test_mdconv(self, dtype=torch.float):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import ModulatedDeformConv2dPack
        input = torch.tensor(input_t).cuda().type(dtype)
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

    def test_mdconv(self):
        self._test_mdconv(torch.double)
        self._test_mdconv(torch.float)
        self._test_mdconv(torch.half)
