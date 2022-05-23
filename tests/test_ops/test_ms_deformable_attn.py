# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttention, MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch)

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck
    _USING_PARROTS = False


@pytest.mark.parametrize('device_type', [
    'cpu',
    pytest.param(
        'cuda:0',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support'))
])
def test_multiscale_deformable_attention(device_type):

    with pytest.raises(ValueError):
        # embed_dims must be divisible by num_heads,
        MultiScaleDeformableAttention(
            embed_dims=256,
            num_heads=7,
        )
    device = torch.device(device_type)
    msda = MultiScaleDeformableAttention(
        embed_dims=3, num_levels=2, num_heads=3)
    msda.init_weights()
    num_query = 5
    bs = 1
    embed_dims = 3
    query = torch.rand(num_query, bs, embed_dims).to(device)
    key = torch.rand(num_query, bs, embed_dims).to(device)
    spatial_shapes = torch.Tensor([[2, 2], [1, 1]]).long().to(device)
    level_start_index = torch.Tensor([0, 4]).long().to(device)
    reference_points = torch.rand(bs, num_query, 2, 2).to(device)
    msda.to(device)
    msda(
        query,
        key,
        key,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index)


def test_forward_multi_scale_deformable_attn_pytorch():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)

    multi_scale_deformable_attn_pytorch(value.double(), shapes,
                                        sampling_locations.double(),
                                        attention_weights.double()).detach()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_forward_equal_with_pytorch_double():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2
    output_pytorch = multi_scale_deformable_attn_pytorch(
        value.double(), shapes, sampling_locations.double(),
        attention_weights.double()).detach().cpu()

    output_cuda = MultiScaleDeformableAttnFunction.apply(
        value.double(), shapes, level_start_index, sampling_locations.double(),
        attention_weights.double(), im2col_step).detach().cpu()
    assert torch.allclose(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    assert max_abs_err < 1e-18
    assert max_rel_err < 1e-15


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_forward_equal_with_pytorch_float():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2
    output_pytorch = multi_scale_deformable_attn_pytorch(
        value, shapes, sampling_locations, attention_weights).detach().cpu()

    output_cuda = MultiScaleDeformableAttnFunction.apply(
        value, shapes, level_start_index, sampling_locations,
        attention_weights, im2col_step).detach().cpu()
    assert torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    assert max_abs_err < 1e-9
    assert max_rel_err < 1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
@pytest.mark.parametrize('channels', [
    4,
    30,
    32,
    64,
    71,
    1025,
])
def test_gradient_numerical(channels,
                            grad_value=True,
                            grad_sampling_loc=True,
                            grad_attn_weight=True):

    N, M, _ = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(3, 2), (2, 1)], dtype=torch.long).cuda()
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    value = torch.rand(N, S, M, channels).cuda() * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).cuda()
    attention_weights = torch.rand(N, Lq, M, L, P).cuda() + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2

    func = MultiScaleDeformableAttnFunction.apply

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight
    if _USING_PARROTS:
        assert gradcheck(
            func, (value.double(), shapes, level_start_index,
                   sampling_locations.double(), attention_weights.double(),
                   im2col_step),
            no_grads=[shapes, level_start_index])
    else:
        assert gradcheck(func, (value.double(), shapes, level_start_index,
                                sampling_locations.double(),
                                attention_weights.double(), im2col_step))
