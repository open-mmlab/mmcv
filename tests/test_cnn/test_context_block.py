import pytest
import torch

from mmcv.cnn.bricks import ContextBlock


def test_context_block():
    with pytest.raises(AssertionError):
        # pooling_type should be in ['att', 'avg']
        ContextBlock(16, 1. / 4, pooling_type='unsupport_type')

    with pytest.raises(AssertionError):
        # fusion_types should be of type list or tuple
        ContextBlock(16, 1. / 4, fusion_types='unsupport_type')

    with pytest.raises(AssertionError):
        # fusion_types should be in ['channel_add', 'channel_mul']
        ContextBlock(16, 1. / 4, fusion_types=('unsupport_type', ))

    # test pooling_type='att'
    imgs = torch.randn(2, 16, 20, 20)
    context_block = ContextBlock(16, 1. / 4, pooling_type='att')
    out = context_block(imgs)
    assert context_block.conv_mask.in_channels == 16
    assert context_block.conv_mask.out_channels == 1
    assert out.shape == imgs.shape

    # test pooling_type='avg'
    imgs = torch.randn(2, 16, 20, 20)
    context_block = ContextBlock(16, 1. / 4, pooling_type='avg')
    out = context_block(imgs)
    assert hasattr(context_block, 'avg_pool')
    assert out.shape == imgs.shape

    # test fusion_types=('channel_add',)
    imgs = torch.randn(2, 16, 20, 20)
    context_block = ContextBlock(16, 1. / 4, fusion_types=('channel_add', ))
    out = context_block(imgs)
    assert context_block.channel_add_conv is not None
    assert context_block.channel_mul_conv is None
    assert out.shape == imgs.shape

    # test fusion_types=('channel_mul',)
    imgs = torch.randn(2, 16, 20, 20)
    context_block = ContextBlock(16, 1. / 4, fusion_types=('channel_mul', ))
    out = context_block(imgs)
    assert context_block.channel_add_conv is None
    assert context_block.channel_mul_conv is not None
    assert out.shape == imgs.shape

    # test fusion_types=('channel_add', 'channel_mul')
    imgs = torch.randn(2, 16, 20, 20)
    context_block = ContextBlock(
        16, 1. / 4, fusion_types=('channel_add', 'channel_mul'))
    out = context_block(imgs)
    assert context_block.channel_add_conv is not None
    assert context_block.channel_mul_conv is not None
    assert out.shape == imgs.shape
