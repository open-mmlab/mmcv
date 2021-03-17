import torch
import MultiScaleDeformableAttention as MSDA
from torch.autograd.function import once_differentiable, Function
from mmcv._ext import

class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------
    # Deformable DETR
    # Copyright (c) 2020 SenseTime. All Rights Reserved.
    # Licensed under the Apache License, Version 2.0 [see LICENSE for details]
    # ------------------------------------------------------------------------------------------------
    # Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
    # ------------------------------------------------------------------------------------------------

    from __future__ import absolute_import
    from __future__ import print_function
    from __future__ import division

    import time
    import torch
    import torch.nn as nn
    from torch.autograd import gradcheck

    from functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch

    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum([(H * W).item() for H, W in shapes])

    torch.manual_seed(3)


    @torch.no_grad()
    def check_forward_equal_with_pytorch_double():
        value = torch.rand(N, S, M, D).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 2
        output_pytorch = ms_deform_attn_core_pytorch(value.double(), shapes, sampling_locations.double(),
                                                     attention_weights.double()).detach().cpu()
        output_cuda = MSDeformAttnFunction.apply(value.double(), shapes, level_start_index, sampling_locations.double(),
                                                 attention_weights.double(), im2col_step).detach().cpu()
        fwdok = torch.allclose(output_cuda, output_pytorch)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

        print(
            f'* {fwdok} check_forward_equal_with_pytorch_double: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


    @torch.no_grad()
    def check_forward_equal_with_pytorch_float():
        value = torch.rand(N, S, M, D).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 2
        output_pytorch = ms_deform_attn_core_pytorch(value, shapes, sampling_locations,
                                                     attention_weights).detach().cpu()
        output_cuda = MSDeformAttnFunction.apply(value, shapes, level_start_index, sampling_locations,
                                                 attention_weights, im2col_step).detach().cpu()
        fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

        print(
            f'* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}')


    def check_gradient_numerical(channels=4, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):

        value = torch.rand(N, S, M, channels).cuda() * 0.01
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
        attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 2
        func = MSDeformAttnFunction.apply

        value.requires_grad = grad_value
        sampling_locations.requires_grad = grad_sampling_loc
        attention_weights.requires_grad = grad_attn_weight

        gradok = gradcheck(func, (
        value.double(), shapes, level_start_index, sampling_locations.double(), attention_weights.double(),
        im2col_step))

        print(f'* {gradok} check_gradient_numerical(D={channels})')


    if __name__ == '__main__':
        check_forward_equal_with_pytorch_double()
        check_forward_equal_with_pytorch_float()

        for channels in [30, 32, 64, 71, 1025, 2048, 3096]:
            check_gradient_numerical(channels, True, True, True)



