# Copyright (c) OpenMMLab. All rights reserved.
# * Modified from
# *https://github.com/SHI-Labs/NATTEN/blob/main/tests/test_na2d.py
#
import pytest
import torch

from mmcv.ops import NeighborhoodAttention


def _priv_test_allclose_cuda(
    batch_size,
    height,
    width,
    kernel_sizes=[3, 5, 7, 9],
    dims=[4, 8],
    heads=[1, 2, 3],
    tol=1e-8,
    device='cuda',
    dtype=torch.float32,
):
    for kernel_size in kernel_sizes:
        for dim in dims:
            for num_heads in heads:
                for qkv_bias in [True, False]:
                    model_kwargs = {
                        'dim': dim * num_heads,
                        'kernel_size': kernel_size,
                        'num_heads': num_heads,
                        'qkv_bias': qkv_bias,
                    }

                    base_state_dict = NeighborhoodAttention(
                        **model_kwargs).state_dict()

                    x = torch.randn((batch_size, height, width,
                                     dim * num_heads)).to(device)

                    nat = NeighborhoodAttention(
                        **model_kwargs).to(device).eval()
                    nat.load_state_dict(base_state_dict, strict=False)

                    y = nat(x)
                    y.sum().backward()


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='requires CUDA support'))
])
@pytest.mark.parametrize('dtype', [torch.float, torch.half])
def test_neighborhood_attention(device, dtype):
    b, li, lj = 4, 14, 16
    _priv_test_allclose_cuda(b, li, lj, device=device, dtype=dtype)
