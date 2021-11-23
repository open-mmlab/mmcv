# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops import Correlation

_input1 = [[[[1., 2., 3.], [0., 1., 2.], [3., 5., 2.]]]]
_input2 = [[[[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]]]]

gt_out_shape = (1, 1, 1, 3, 3)
_gt_out = [[[[[1., 4., 9.], [0., 1., 4.], [24., 25., 4.]]]]]
gt_input1_grad = [[[[1., 2., 3.], [3., 1., 2.], [8., 5., 2.]]]]


def assert_equal_tensor(tensor_a, tensor_b):

    assert tensor_a.eq(tensor_b).all()


class TestCorrelation(object):

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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA support')
    @pytest.mark.parametrize('dtype', [torch.float, torch.double, torch.half])
    def test_correlation(self, dtype):
        self._test_correlation(dtype=dtype)
