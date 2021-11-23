import pytest
import torch
from torch.autograd import gradcheck

from mmcv.ops import CARAFE, CARAFENaive


class TestCarafe(object):

    def setup_class(self):
        self.feat = torch.randn(
            2, 64, 3, 3, requires_grad=True, device='cuda').double()
        self.mask = torch.randn(
            2, 100, 6, 6, requires_grad=True,
            device='cuda').sigmoid().double()
        self._CARAFE = {'naive': CARAFENaive, 'plain': CARAFE}

    @pytest.mark.parametrize('mode', ['naive', 'plain'])
    def test_carafe_naive_gradcheck(self, mode):
        if not torch.cuda.is_available():
            return
        gradcheck(
            self._CARAFE[mode](5, 4, 2), (self.feat, self.mask),
            atol=1e-4,
            eps=1e-4)
