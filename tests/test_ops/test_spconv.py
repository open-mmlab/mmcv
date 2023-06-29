# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch import nn

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.ops import (SparseConvTensor, SparseInverseConv3d, SparseSequential,
                      SubMConv3d)

if torch.__version__ == 'parrots':
    pytest.skip('not supported in parrots now', allow_module_level=True)

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE


def make_sparse_convmodule(in_channels,
                           out_channels,
                           kernel_size,
                           indice_key,
                           stride=1,
                           padding=0,
                           conv_type='SubMConv3d',
                           norm_cfg=None,
                           order=('conv', 'norm', 'act')):
    """Make sparse convolution module.

    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of out channels
        kernel_size (int|tuple(int)): kernel size of convolution
        indice_key (str): the indice key used for sparse tensor
        stride (int|tuple(int)): the stride of convolution
        padding (int or list[int]): the padding number of input
        conv_type (str): sparse conv type in spconv
        norm_cfg (dict[str]): config of normalization layer
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").

    Returns:
        spconv.SparseSequential: sparse convolution module.
    """
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}

    conv_cfg = dict(type=conv_type, indice_key=indice_key)

    layers = list()
    for layer in order:
        if layer == 'conv':
            if conv_type not in [
                    'SparseInverseConv3d', 'SparseInverseConv2d',
                    'SparseInverseConv1d'
            ]:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=stride,
                        padding=padding,
                        bias=False))
            else:
                layers.append(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size,
                        bias=False))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))

    layers = SparseSequential(*layers)
    return layers


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'mlu',
        marks=pytest.mark.skipif(
            not IS_MLU_AVAILABLE, reason='requires MLU support'))
])
def test_make_sparse_convmodule(device):
    torch.cuda.empty_cache()
    voxel_features = torch.tensor([[6.56126, 0.9648336, -1.7339306, 0.315],
                                   [6.8162713, -2.480431, -1.3616394, 0.36],
                                   [11.643568, -4.744306, -1.3580885, 0.16],
                                   [23.482342, 6.5036807, 0.5806964, 0.35]],
                                  dtype=torch.float32,
                                  device=device)  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32,
        device=device)  # n, 4(batch, ind_x, ind_y, ind_z)

    # test
    input_sp_tensor = SparseConvTensor(voxel_features, coordinates,
                                       [41, 1600, 1408], 2)

    sparse_block0 = make_sparse_convmodule(
        4,
        16,
        3,
        'test0',
        stride=1,
        padding=0,
        conv_type='SubMConv3d',
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        order=('conv', 'norm', 'act')).to(device)
    assert isinstance(sparse_block0[0], SubMConv3d)
    assert sparse_block0[0].in_channels == 4
    assert sparse_block0[0].out_channels == 16
    assert isinstance(sparse_block0[1], torch.nn.BatchNorm1d)
    assert sparse_block0[1].eps == 0.001
    assert sparse_block0[1].momentum == 0.01
    assert isinstance(sparse_block0[2], torch.nn.ReLU)

    # test forward
    out_features = sparse_block0(input_sp_tensor)
    assert out_features.features.shape == torch.Size([4, 16])

    # device == mlu: not support inverse==1 yet
    if device != 'mlu':
        sparse_block1 = make_sparse_convmodule(
            4,
            16,
            3,
            'test1',
            stride=1,
            padding=0,
            conv_type='SparseInverseConv3d',
            norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
            order=('norm', 'act', 'conv')).to(device)
        assert isinstance(sparse_block1[2], SparseInverseConv3d)
        assert isinstance(sparse_block1[0], torch.nn.BatchNorm1d)
        assert isinstance(sparse_block1[1], torch.nn.ReLU)
