import torch

from mmcv.op import MaskedConv2d


class TestMaskedConv2d(object):

    def test_masked_conv2d(self):
        input = torch.randn(1, 3, 16, 16, requires_grad=True, device='cuda')
        mask = torch.randn(1, 16, 16, requires_grad=True, device='cuda')
        conv = MaskedConv2d(3, 3, 3).cuda()
        output = conv(input, mask)
        assert output is not None
