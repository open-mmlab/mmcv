# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmcv.cnn import NonLocal1d, NonLocal2d, NonLocal3d
from mmcv.cnn.bricks.non_local import _NonLocalNd


def test_nonlocal():
    with pytest.raises(ValueError):
        # mode should be in ['embedded_gaussian', 'dot_product']
        _NonLocalNd(3, mode='unsupport_mode')

    # _NonLocalNd with zero initialization
    _NonLocalNd(3)
    _NonLocalNd(3, norm_cfg=dict(type='BN'))

    # _NonLocalNd without zero initialization
    _NonLocalNd(3, zeros_init=False)
    _NonLocalNd(3, norm_cfg=dict(type='BN'), zeros_init=False)


def test_nonlocal3d():
    # NonLocal3d with 'embedded_gaussian' mode
    imgs = torch.randn(2, 3, 10, 20, 20)
    nonlocal_3d = NonLocal3d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            # NonLocal is only implemented on gpu in parrots
            imgs = imgs.cuda()
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    # NonLocal3d with 'dot_product' mode
    nonlocal_3d = NonLocal3d(3, mode='dot_product')
    assert nonlocal_3d.mode == 'dot_product'
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    # NonLocal3d with 'concatenation' mode
    nonlocal_3d = NonLocal3d(3, mode='concatenation')
    assert nonlocal_3d.mode == 'concatenation'
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    # NonLocal3d with 'gaussian' mode
    nonlocal_3d = NonLocal3d(3, mode='gaussian')
    assert not hasattr(nonlocal_3d, 'phi')
    assert nonlocal_3d.mode == 'gaussian'
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    # NonLocal3d with 'gaussian' mode and sub_sample
    nonlocal_3d = NonLocal3d(3, mode='gaussian', sub_sample=True)
    assert isinstance(nonlocal_3d.g, nn.Sequential) and len(nonlocal_3d.g) == 2
    assert isinstance(nonlocal_3d.g[1], nn.MaxPool3d)
    assert nonlocal_3d.g[1].kernel_size == (1, 2, 2)
    assert isinstance(nonlocal_3d.phi, nn.MaxPool3d)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    # NonLocal3d with 'dot_product' mode and sub_sample
    nonlocal_3d = NonLocal3d(3, mode='dot_product', sub_sample=True)
    for m in [nonlocal_3d.g, nonlocal_3d.phi]:
        assert isinstance(m, nn.Sequential) and len(m) == 2
        assert isinstance(m[1], nn.MaxPool3d)
        assert m[1].kernel_size == (1, 2, 2)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape


def test_nonlocal2d():
    # NonLocal2d with 'embedded_gaussian' mode
    imgs = torch.randn(2, 3, 20, 20)
    nonlocal_2d = NonLocal2d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

    # NonLocal2d with 'dot_product' mode
    imgs = torch.randn(2, 3, 20, 20)
    nonlocal_2d = NonLocal2d(3, mode='dot_product')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

    # NonLocal2d with 'concatenation' mode
    imgs = torch.randn(2, 3, 20, 20)
    nonlocal_2d = NonLocal2d(3, mode='concatenation')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

    # NonLocal2d with 'gaussian' mode
    imgs = torch.randn(2, 3, 20, 20)
    nonlocal_2d = NonLocal2d(3, mode='gaussian')
    assert not hasattr(nonlocal_2d, 'phi')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

    # NonLocal2d with 'gaussian' mode and sub_sample
    nonlocal_2d = NonLocal2d(3, mode='gaussian', sub_sample=True)
    assert isinstance(nonlocal_2d.g, nn.Sequential) and len(nonlocal_2d.g) == 2
    assert isinstance(nonlocal_2d.g[1], nn.MaxPool2d)
    assert nonlocal_2d.g[1].kernel_size == (2, 2)
    assert isinstance(nonlocal_2d.phi, nn.MaxPool2d)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

    # NonLocal2d with 'dot_product' mode and sub_sample
    nonlocal_2d = NonLocal2d(3, mode='dot_product', sub_sample=True)
    for m in [nonlocal_2d.g, nonlocal_2d.phi]:
        assert isinstance(m, nn.Sequential) and len(m) == 2
        assert isinstance(m[1], nn.MaxPool2d)
        assert m[1].kernel_size == (2, 2)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape


def test_nonlocal1d():
    # NonLocal1d with 'embedded_gaussian' mode
    imgs = torch.randn(2, 3, 20)
    nonlocal_1d = NonLocal1d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

    # NonLocal1d with 'dot_product' mode
    imgs = torch.randn(2, 3, 20)
    nonlocal_1d = NonLocal1d(3, mode='dot_product')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

    # NonLocal1d with 'concatenation' mode
    imgs = torch.randn(2, 3, 20)
    nonlocal_1d = NonLocal1d(3, mode='concatenation')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

    # NonLocal1d with 'gaussian' mode
    imgs = torch.randn(2, 3, 20)
    nonlocal_1d = NonLocal1d(3, mode='gaussian')
    assert not hasattr(nonlocal_1d, 'phi')
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

    # NonLocal1d with 'gaussian' mode and sub_sample
    nonlocal_1d = NonLocal1d(3, mode='gaussian', sub_sample=True)
    assert isinstance(nonlocal_1d.g, nn.Sequential) and len(nonlocal_1d.g) == 2
    assert isinstance(nonlocal_1d.g[1], nn.MaxPool1d)
    assert nonlocal_1d.g[1].kernel_size == 2
    assert isinstance(nonlocal_1d.phi, nn.MaxPool1d)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

    # NonLocal1d with 'dot_product' mode and sub_sample
    nonlocal_1d = NonLocal1d(3, mode='dot_product', sub_sample=True)
    for m in [nonlocal_1d.g, nonlocal_1d.phi]:
        assert isinstance(m, nn.Sequential) and len(m) == 2
        assert isinstance(m[1], nn.MaxPool1d)
        assert m[1].kernel_size == 2
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape
