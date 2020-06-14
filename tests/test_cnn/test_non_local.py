import pytest
import torch
import torch.nn as nn

from mmcv.cnn import NonLocal1d, NonLocal2d, NonLocal3d
from mmcv.cnn.bricks.non_local import _NonLocalNd


def test_nonlocal():
    with pytest.raises(ValueError):
        # mode should be in ['embedded_gaussian', 'dot_product']
        _NonLocalNd(3, mode='unsupport_mode')

    # _NonLocalNd
    _NonLocalNd(3, norm_cfg=dict(type='BN'))
    # Not Zero initialization
    _NonLocalNd(3, norm_cfg=dict(type='BN'), zeros_init=True)

    # NonLocal3d
    imgs = torch.randn(2, 3, 10, 20, 20)
    nonlocal_3d = NonLocal3d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            # NonLocal is only implemented on gpu in parrots
            imgs = imgs.cuda()
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

    nonlocal_3d = NonLocal3d(3, mode='dot_product')
    assert nonlocal_3d.mode == 'dot_product'
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            nonlocal_3d.cuda()
    out = nonlocal_3d(imgs)
    assert out.shape == imgs.shape

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

    # NonLocal2d
    imgs = torch.randn(2, 3, 20, 20)
    nonlocal_2d = NonLocal2d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_2d.cuda()
    out = nonlocal_2d(imgs)
    assert out.shape == imgs.shape

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

    # NonLocal1d
    imgs = torch.randn(2, 3, 20)
    nonlocal_1d = NonLocal1d(3)
    if torch.__version__ == 'parrots':
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            nonlocal_1d.cuda()
    out = nonlocal_1d(imgs)
    assert out.shape == imgs.shape

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
