import torch
from torch.autograd import gradcheck


class TestCornerPool(object):

    def test_corner_pool_top_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import CornerPool
        input = torch.randn(2, 4, 5, 5, requires_grad=True, device='cuda')
        gradcheck(CornerPool('top'), (input, ), atol=1e-3, eps=1e-4)

    def test_corner_pool_bottom_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import CornerPool
        input = torch.randn(2, 4, 5, 5, requires_grad=True, device='cuda')
        gradcheck(CornerPool('bottom'), (input, ), atol=1e-3, eps=1e-4)

    def test_corner_pool_left_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import CornerPool
        input = torch.randn(2, 4, 5, 5, requires_grad=True, device='cuda')
        gradcheck(CornerPool('left'), (input, ), atol=1e-3, eps=1e-4)

    def test_corner_pool_right_gradcheck(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import CornerPool
        input = torch.randn(2, 4, 5, 5, requires_grad=True, device='cuda')
        gradcheck(CornerPool('right'), (input, ), atol=1e-3, eps=1e-4)
