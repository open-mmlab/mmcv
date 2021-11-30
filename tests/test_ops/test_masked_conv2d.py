import pytest
import torch

from mmcv.ops import MaskedConv2d


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
class TestMaskedConv2d(object):

    def test_masked_conv2d(self):
        input = torch.randn(1, 3, 16, 16, requires_grad=True, device='cuda')
        mask = torch.randn(1, 16, 16, requires_grad=True, device='cuda')
        conv = MaskedConv2d(3, 3, 3).cuda()
        output = conv(input, mask)
        assert output is not None
