"""
CommandLine:
    pytest tests/test_corner_pool.py
"""
import pytest
import torch

from mmcv.ops import CornerPool


class TestCornerPool(object):

    def setup_class(self):
        self.lr_tensor = torch.tensor([[[[0, 0, 0, 0, 0], [2, 1, 3, 0, 2],
                                         [5, 4, 1, 1, 6], [0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0]]]])
        self.tb_tensor = torch.tensor([[[[0, 3, 1, 0, 0], [0, 1, 1, 0, 0],
                                         [0, 3, 4, 0, 0], [0, 2, 2, 0, 0],
                                         [0, 0, 2, 0, 0]]]])
        self.left_answer = torch.tensor([[[[0, 0, 0, 0, 0], [3, 3, 3, 2, 2],
                                           [6, 6, 6, 6, 6], [0, 0, 0, 0, 0],
                                           [0, 0, 0, 0, 0]]]])
        self.right_answer = torch.tensor([[[[0, 0, 0, 0, 0], [2, 2, 3, 3, 3],
                                            [5, 5, 5, 5, 6], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]]])
        self.top_answer = torch.tensor([[[[0, 3, 4, 0, 0], [0, 3, 4, 0, 0],
                                          [0, 3, 4, 0, 0], [0, 2, 2, 0, 0],
                                          [0, 0, 2, 0, 0]]]])
        self.bottom_answer = torch.tensor([[[[0, 3, 1, 0, 0], [0, 3, 1, 0, 0],
                                             [0, 3, 4, 0, 0], [0, 3, 4, 0, 0],
                                             [0, 3, 4, 0, 0]]]])
        self.input = {
            'left': self.lr_tensor,
            'right': self.lr_tensor,
            'top': self.tb_tensor,
            'bottom': self.tb_tensor
        }
        self.answer = {
            'left': self.left_answer,
            'right': self.right_answer,
            'top': self.top_answer,
            'bottom': self.bottom_answer
        }

    @pytest.mark.parametrize('mode',
                             ['corner', 'left', 'right', 'top', 'bottom'])
    def test_corner_pool_device_and_dtypes_cpu(self, mode):
        """
        CommandLine:
            xdoctest -m tests/test_corner_pool.py \
                test_corner_pool_device_and_dtypes_cpu
        """
        if mode == 'corner':
            with pytest.raises(AssertionError):
                # pool mode must in ['bottom', 'left', 'right', 'top']
                pool = CornerPool('corner')
        else:
            pool = CornerPool(mode)
            output = pool(self.input[mode])
            assert output.type() == self.input[mode].type()
            assert torch.equal(output, self.answer[mode])
