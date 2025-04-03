# Copyright (c) OpenMMLab. All rights reserved.
# * Modified from
# *https://github.com/SHI-Labs/NATTEN/blob/main/tests/test_na2d.py
#
import pytest
import torch

from mmcv.ops import NeighborhoodAttention


def _test_allclose_cuda(
    batch_size,
    height,
    width,
    kernel_size,
    dim,
    num_heads,
    qkv_bias,
    device='cuda',
    dtype=torch.float32,
):
    model_kwargs = {
        'dim': dim * num_heads,
        'kernel_size': kernel_size,
        'num_heads': num_heads,
        'qkv_bias': qkv_bias,
    }
    base_state_dict = NeighborhoodAttention(**model_kwargs).state_dict()
    x = torch.randn((batch_size, height, width, dim * num_heads),
                    dtype=dtype).to(device)
    nat = NeighborhoodAttention(**model_kwargs).to(device).eval()
    if dtype == torch.half:
        nat = nat.half()
    nat.load_state_dict(base_state_dict, strict=False)
    y = nat(x)
    y.sum().backward()


@pytest.mark.parametrize('kernel_size', [3, 5, 7, 9])
@pytest.mark.parametrize('dim', [4, 8])
@pytest.mark.parametrize('num_heads', [1, 2, 3])
@pytest.mark.parametrize('qkv_bias', [True, False])
@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support'))
])
@pytest.mark.parametrize('dtype', [torch.float, torch.half])
def test_neighborhood_attention(kernel_size, dim, num_heads, qkv_bias, device,
                                dtype):
    b, li, lj = 4, 14, 16
    _test_allclose_cuda(
        b,
        li,
        lj,
        kernel_size=kernel_size,
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        device=device,
        dtype=dtype)
