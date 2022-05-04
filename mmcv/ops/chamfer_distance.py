# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['chamfer_distance_forward', 'chamfer_distance_backward'])


class ChamferDistanceFunction(Function):
    """This is an implementation of the 2D Chamfer Distance.

    It has been used in the paper `Oriented RepPoints for Aerial Object
    Detection (CVPR 2022) <https://arxiv.org/abs/2105.11111>_`.
    """

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        """
        Args:
            xyz1 (torch.Tensor): Point set with shape (B, N, 2).
            xyz2 (torch.Tensor): Point set with shape (B, N, 2).

        Returns:
            tuple:

                - dist1 (torch.Tensor): Chamfer ditacne (xyz1 to xyz2) with
                    shape (B, N).
                - dist2 (torch.Tensor): Chamfer ditacne (xyz2 to xyz1) with
                    shape (B, N).
                - idx1 (torch.Tensor): Index of chamfer ditacne (xyz1 to xyz2)
                    with shape (B, N), which be used in compute gradient.
                - idx2 (torch.Tensor): Index of chamfer ditacne (xyz2 to xyz2)
                    with shape (B, N), which be used in compute gradient.
        """
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        device = xyz1.device

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)
        torch.cuda.set_device(device)

        ext_module.chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1,
                                            idx2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    @once_differentiable
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        """

        Args:
            graddist1 (torch.Tensor): Gradient of chamfer ditacne
                (xyz1 to xyz2) with shape (B, N).
            graddist2 (torch.Tensor): Gradient of chamfer ditacne
                (xyz2 to xyz1) with shape (B, N).
            gradidx1 (torch.Tensor): Index of chamfer ditacne (xyz1 to xyz2)
                    with shape (B, N), which be used in compute gradient.
            gradidx2 (torch.Tensor): Index of chamfer ditacne (xyz2 to xyz2)
                    with shape (B, N), which be used in compute gradient.

        Returns:
            tuple:

            - gradxyz1 (torch.Tensor): Gradient of the point set with shape \
                (B, N, 2).
            - gradxyz2 (torch.Tensor):Gradient of the point set with shape \
                (B, N, 2).
        """
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        device = graddist1.device

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)
        ext_module.chamfer_distance_backward(xyz1, xyz2, gradxyz1, gradxyz2,
                                             graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDistance(nn.Module):
    """This is an implementation of the 2D Chamfer Distance."""

    def forward(self, input1, input2):
        """
        Args:
            input1 (torch.Tensor): Point set with shape (B, N, 2).
            input2 (torch.Tensor): Point set with shape (B, N, 2).
        """
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return ChamferDistanceFunction.apply(input1, input2)