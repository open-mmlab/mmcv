import torch
from torch.autograd import gradcheck


class TestCarafe(object):

    def test_carafe_naive_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFENaive
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFENaive(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)

    def test_carafe_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import CARAFE
        feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        gradcheck(CARAFE(5, 4, 2), (feat, mask), atol=1e-4, eps=1e-4)
