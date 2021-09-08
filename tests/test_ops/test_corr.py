# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import AllPairsCorrelation, Correlation

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
_ap_grad_gt1 = [[[[27., 27., 27.], [27., 27., 27.], [27., 27., 27.]]]]
_ap_grad_gt2 = [[[[19., 19., 19.], [19., 19., 19.], [19., 19., 19.]]]]
_ap_gt_out2 = [[[[[1., 2.], [3., 1.], [8., 5.]],
                 [[2., 4.], [6., 2.], [16., 10.]],
                 [[3., 6.], [9., 3.], [24., 15.]]],
                [[[0., 0.], [0., 0.], [0., 0.]], [[1., 2.], [3., 1.], [8.,
                                                                       5.]],
                 [[2., 4.], [6., 2.], [16., 10.]]],
                [[[3., 6.], [9., 3.], [24., 15.]],
                 [[5., 10.], [15., 5.], [40., 25.]],
                 [[2., 4.], [6., 2.], [16., 10.]]]]]
_ap_grad_gt2_2 = [[[[19., 19.], [19., 19.], [19., 19.]]]]


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


class TestAllPairsCorrelation:

    def _test_all_pairs_correlation(self, dtype=torch.float):
        layer = AllPairsCorrelation()
        input1 = torch.tensor(_input1, dtype=dtype).cuda()
        input2 = torch.tensor(_input2, dtype=dtype).cuda()
        input2_2 = torch.tensor(_input2_2, dtype=dtype).cuda()
        input1.requires_grad = True
        input2.requires_grad = True
        input2_2.requires_grad = True

        ap_gt_out = torch.tensor(_ap_gt_out, dtype=dtype)
        ap_grad_gt1 = torch.tensor(_ap_grad_gt1, dtype=dtype)
        ap_grad_gt2 = torch.tensor(_ap_grad_gt2, dtype=dtype)
        ap_grad_gt2_2 = torch.tensor(_ap_grad_gt2_2, dtype=dtype)

        out = layer(input1, input2)
        out.backward(torch.ones_like(out))
        assert_equal_tensor(out.cpu(), ap_gt_out)
        assert_equal_tensor(input1.grad.detach().cpu(), ap_grad_gt1)
        assert_equal_tensor(input2.grad.detach().cpu(), ap_grad_gt2)

        out2 = layer(input1, input2_2)
        out2.backward(torch.ones_like(out2))
        ap_gt_out2 = torch.tensor(_ap_gt_out2, dtype=dtype)
        assert_equal_tensor(out2.cpu(), ap_gt_out2)
        assert_equal_tensor(input2_2.grad.detach().cpu(), ap_grad_gt2_2)

    def test_all_pairs_correlation(self):
        self._test_all_pairs_correlation(torch.float)
        self._test_all_pairs_correlation(torch.double)
        self._test_all_pairs_correlation(torch.half)
