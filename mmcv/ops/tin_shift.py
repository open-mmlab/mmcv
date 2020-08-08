# Code reference from "Temporal Interlacing Network"
# https://github.com/deepcs233/TIN/blob/master/cuda_shift/rtc_wrap.py
# Hao Shao, Shengju Qian, Yu Liu
# shaoh19@mails.tsinghua.edu.cn, sjqian@cse.cuhk.edu.hk, yuliu@ee.cuhk.edu.hk

import torch
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext',
                                 ['tin_shift_forward', 'tin_shift_backward'])


class TINShiftFunction(Function):

    @staticmethod
    def forward(ctx, input, shift):

        if input.requires_grad:
            ctx.save_for_backward(shift)

        out = torch.zeros_like(input)
        ext_module.tin_shift_forward(input, shift, out)

        return out

    @staticmethod
    def backward(ctx, grad_output):

        shift = ctx.saved_tensors[0]
        data_grad_input = grad_output.new(*grad_output.size()).zero_()
        shift_grad_input = shift.new(*shift.size()).zero_()
        ext_module.tin_shift_backward(grad_output, shift, data_grad_input)

        return data_grad_input, shift_grad_input
