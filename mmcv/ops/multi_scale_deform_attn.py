import torch
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


class MultiScaleDeformableAttnFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ext_module.ms_deform_attn_forward(value, value_spatial_shapes,
                                                   value_level_start_index,
                                                   sampling_locations,
                                                   attention_weights,
                                                   ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index,\
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            ext_module.ms_deform_attn_backward(
                value,
                value_spatial_shapes,
                value_level_start_index,
                sampling_locations,
                attention_weights,
                grad_output,
                ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes,
                                        sampling_locations, attention_weights):
    """

    Args:
        value (): The value
        value_spatial_shapes (Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent ()
        sampling_locations:
        attention_weights:

    Returns:

    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes],
                             dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ ->
        # N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(
            N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :,
                                          lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()
