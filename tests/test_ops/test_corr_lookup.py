# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.ops import CorrLookup


def test_corr_lookup():
    corr_lookup_module = CorrLookup(4)
    B = 2
    H = 16
    W = 16
    corr1 = torch.ones((2 * H * W, 1, H, W))
    corr2 = torch.ones((2 * H * W, 1, H // 2, W // 2))
    corr_pyramid = [corr1, corr2]

    flow = torch.randn(B, 2, H, W)

    out = corr_lookup_module(corr_pyramid, flow)

    assert out.shape == torch.Size((B, 81 * 2, H, W))
