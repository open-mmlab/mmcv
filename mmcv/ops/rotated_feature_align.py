# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext',
    ['rotated_feature_align_forward', 'rotated_feature_align_backward'])


class RotatedFeatureAlignFunction(Function):
    """Using the feature interpolation to obtain the position information
    correspond to the refined rotate anchors and reconstruct the feature maps
    in pixel-wise manner to achieve feature alignment.

    The details are described in the paper
    `R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating
    Object <https://arxiv.org/abs/1908.05612>`_.
    """

    @staticmethod
    def symbolic(g, features, best_rbboxes, spatial_scale, points):
        assert points in [1, 5]
        return g.op(
            'mmcv::MMCVRotatedFeatureAlign',
            features,
            best_rbboxes,
            spatial_scale_f=spatial_scale,
            points_i=points)

    @staticmethod
    def forward(ctx, features, best_rbboxes, spatial_scale, points):
        """
        Args:
            features (torch.Tensor): Input features with shape [N,C,H,W].
            best_rbboxes (torch.Tensor): Refined rotate anchors with
                shape [N,H,W,5]. Coordinate format (cx,cx,h,w,a).
            spatial_scale (float): The scale of feature map size and
                input image size.
            points (int, optional): The number of sample points.
                Only 1 and 5 are supported. Defaults to 1.

        Returns:
            torch.Tensor: Refined features with shape [N,C,H,W].
        """
        ctx.spatial_scale = spatial_scale
        ctx.points = points
        ctx.save_for_backward(best_rbboxes)
        assert points in [1, 5]
        output = torch.zeros_like(features)
        ext_module.rotated_feature_align_forward(
            features,
            best_rbboxes,
            output,
            spatial_scale=spatial_scale,
            points=points)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Args:
            grad_output (torch.Tensor): The gradiant of output features
                with shape [N,C,H,W].

        Returns:
            torch.Tensor: The gradiant of input features with shape [N,C,H,W].
        """
        best_rbboxes = ctx.saved_tensors[0]
        points = ctx.points
        spatial_scale = ctx.spatial_scale
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(grad_output)
            ext_module.rotated_feature_align_backward(
                grad_output.contiguous(),
                best_rbboxes,
                grad_input,
                spatial_scale=spatial_scale,
                points=points)
        return grad_input, None, None, None


def rotated_feature_align(features,
                          best_rbboxes,
                          spatial_scale=1 / 8,
                          points=1):
    return RotatedFeatureAlignFunction.apply(features, best_rbboxes,
                                             spatial_scale, points)
