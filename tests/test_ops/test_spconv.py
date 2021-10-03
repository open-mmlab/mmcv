import torch

from mmcv.ops import SparseBasicBlock, SparseConvTensor, SubMConv3d


def test_SparseBasicBlock():
    voxel_features = torch.tensor([[6.56126, 0.9648336, -1.7339306, 0.315],
                                   [6.8162713, -2.480431, -1.3616394, 0.36],
                                   [11.643568, -4.744306, -1.3580885, 0.16],
                                   [23.482342, 6.5036807, 0.5806964, 0.35]],
                                  dtype=torch.float32)  # n, point_features
    coordinates = torch.tensor(
        [[0, 12, 819, 131], [0, 16, 750, 136], [1, 16, 705, 232],
         [1, 35, 930, 469]],
        dtype=torch.int32)  # n, 4(batch, ind_x, ind_y, ind_z)

    # test
    input_sp_tensor = SparseConvTensor(voxel_features, coordinates,
                                       [41, 1600, 1408], 2)
    self = SparseBasicBlock(
        4,
        4,
        conv_cfg=dict(type='SubMConv3d', indice_key='subm1'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01))
    # test conv and bn layer
    assert isinstance(self.conv1, SubMConv3d)
    assert self.conv1.in_channels == 4
    assert self.conv1.out_channels == 4
    assert isinstance(self.conv2, SubMConv3d)
    assert self.conv2.out_channels == 4
    assert self.conv2.out_channels == 4
    assert self.bn1.eps == 1e-3
    assert self.bn1.momentum == 0.01

    out_features = self(input_sp_tensor)
    assert out_features.features.shape == torch.Size([4, 4])
