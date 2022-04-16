# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.cnn.bricks import GeneralizedAttention


def test_context_block():

    # test attention_type='1000'
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, attention_type='1000')
    assert gen_attention_block.query_conv.in_channels == 16
    assert gen_attention_block.key_conv.in_channels == 16
    assert gen_attention_block.key_conv.in_channels == 16
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test attention_type='0100'
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, attention_type='0100')
    assert gen_attention_block.query_conv.in_channels == 16
    assert gen_attention_block.appr_geom_fc_x.in_features == 8
    assert gen_attention_block.appr_geom_fc_y.in_features == 8
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test attention_type='0010'
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, attention_type='0010')
    assert gen_attention_block.key_conv.in_channels == 16
    assert hasattr(gen_attention_block, 'appr_bias')
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test attention_type='0001'
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, attention_type='0001')
    assert gen_attention_block.appr_geom_fc_x.in_features == 8
    assert gen_attention_block.appr_geom_fc_y.in_features == 8
    assert hasattr(gen_attention_block, 'geom_bias')
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test spatial_range >= 0
    imgs = torch.randn(2, 256, 20, 20)
    gen_attention_block = GeneralizedAttention(256, spatial_range=10)
    assert hasattr(gen_attention_block, 'local_constraint_map')
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test q_stride > 1
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, q_stride=2)
    assert gen_attention_block.q_downsample is not None
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test kv_stride > 1
    imgs = torch.randn(2, 16, 20, 20)
    gen_attention_block = GeneralizedAttention(16, kv_stride=2)
    assert gen_attention_block.kv_downsample is not None
    out = gen_attention_block(imgs)
    assert out.shape == imgs.shape

    # test fp16 with attention_type='1111'
    if torch.cuda.is_available():
        imgs = torch.randn(2, 16, 20, 20).cuda().to(torch.half)
        gen_attention_block = GeneralizedAttention(
            16,
            spatial_range=-1,
            num_heads=8,
            attention_type='1111',
            kv_stride=2)
        gen_attention_block.cuda().type(torch.half)
        out = gen_attention_block(imgs)
        assert out.shape == imgs.shape
