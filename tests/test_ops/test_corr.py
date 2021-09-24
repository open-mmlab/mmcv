# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import Correlation

_input1 = [[[[1., 2., 3.], [0., 1., 2.], [3., 5., 2.]]]]
_input2 = [[[[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]]]]
_input2_2 = [[[[1., 2.], [3., 1.], [8., 5.]]]]
gt_out_shape = (1, 1, 1, 3, 3)
_gt_out = [[[[[1., 4., 9.], [0., 1., 4.], [24., 25., 4.]]]]]
gt_input1_grad = [[[[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]]]]
_ap_gt_out = [[[[[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]],
                [[2., 4., 6.], [6., 2., 4.], [16., 10., 4.]],
                [[3., 6., 9.], [9., 3., 6.], [24., 15., 6.]]],
               [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
                [[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]],
                [[2., 4., 6.], [6., 2., 4.], [16., 10., 4.]]],
               [[[3., 6., 9.], [9., 3., 6.], [24., 15., 6.]],
                [[5., 10., 15.], [15., 5., 10.], [40., 25., 10.]],
                [[2., 4., 6.], [6., 2., 4.], [16., 10., 4.]]]]]


def assert_equal_tensor(tensor_a, tensor_b):

    assert tensor_a.eq(tensor_b).all()


class TestCorrelation:

    def _test_correlation(self, dtype=torch.float):

        layer = Correlation(max_displacement=0)

        input1 = torch.tensor(_input1, dtype=dtype).cuda()
        input2 = torch.tensor(_input2, dtype=dtype).cuda()
        input1.requires_grad = True
        input2.requires_grad = True
        out = layer(input1, input2)
        out.backward(torch.ones_like(out))

        gt_out = torch.tensor(_gt_out, dtype=dtype)
        assert_equal_tensor(out.cpu(), gt_out)
        assert_equal_tensor(input1.grad.detach().cpu(), input2.cpu())
        assert_equal_tensor(input2.grad.detach().cpu(), input1.cpu())

    def test_correlation(self):
        self._test_correlation(torch.float)
        self._test_correlation(torch.double)
        self._test_correlation(torch.half)
