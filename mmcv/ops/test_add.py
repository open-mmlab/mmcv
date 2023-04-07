# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.autograd import Function

from ..utils import ext_loader


ext_module = ext_loader.load_ext('_ext', [
    'test_add_forward'
])


class TestAdd(Function):
    @staticmethod
    def symbolic(g, input1, input2):
        return g.op(
            'mmcv::TestAdd',
            input1,
            input2)

    @staticmethod
    def forward(
            ctx,
            input1: torch.Tensor,
            input2: torch.Tensor) -> torch.Tensor:
   
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        output = input1.new_zeros(input1.shape)
        ext_module.test_add_forward(
            input1,
            input2,
            output,
        )
        return output

test_add = TestAdd.apply
