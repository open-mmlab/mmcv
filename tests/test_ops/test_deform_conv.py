import numpy as np
import torch

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


class TestDeformconv(object):

    def _test_deformconv(self, dtype=torch.float, threshold=1e-3):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import DeformConv2dPack
        c_in = 1
        c_out = 1
        x = torch.Tensor(input).cuda().type(dtype)
        x.requires_grad = True
        model = DeformConv2dPack(c_in, c_out, 2, stride=1, padding=0)
        model.conv_offset.weight.data = torch.nn.Parameter(
            torch.Tensor(offset_weight).reshape(8, 1, 2, 2))
        model.conv_offset.bias.data = torch.nn.Parameter(
            torch.Tensor(offset_bias).reshape(8))
        model.weight.data = torch.nn.Parameter(
            torch.Tensor(deform_weight).reshape(1, 1, 2, 2))
        model.cuda().type(dtype)

        out = model(x)
        out.backward(torch.ones_like(out))

        assert np.allclose(out.data.detach().cpu().numpy(), gt_out, threshold)
        assert np.allclose(x.grad.detach().cpu().numpy(), gt_x_grad, threshold)
        assert np.allclose(
            model.conv_offset.weight.grad.detach().cpu().numpy(),
            gt_offset_weight_grad, threshold)
        assert np.allclose(model.conv_offset.bias.grad.detach().cpu().numpy(),
                           gt_offset_bias_grad, threshold)
        assert np.allclose(model.weight.grad.detach().cpu().numpy(),
                           gt_deform_weight_grad, threshold)

    def test_deformconv(self):
        self._test_deformconv(torch.double)
        self._test_deformconv(torch.float)
        self._test_deformconv(torch.half, 1e-1)
