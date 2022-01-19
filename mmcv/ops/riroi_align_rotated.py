# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader, is_tuple_of

ext_module = ext_loader.load_ext(
    '_ext', ['riroi_align_rotated_forward', 'riroi_align_rotated_backward'])


class RiRoIAlignRotatedFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                num_samples=0,
                num_orientations=8,
                clockwise=False):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif is_tuple_of(out_size, int):
            assert len(out_size) == 2
            out_h, out_w = out_size
        else:
            raise TypeError(
                f'"out_size" should be an integer or tuple of integers,'
                f' but got {out_size}')
        ctx.spatial_scale = spatial_scale
        ctx.num_samples = num_samples
        ctx.num_orientations = num_orientations
        ctx.clockwise = clockwise
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, _, _ = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)

        ext_module.riroi_align_rotated_forward(
            features,
            rois,
            output,
            pooled_height=out_h,
            pooled_width=out_w,
            spatial_scale=spatial_scale,
            num_samples=num_samples,
            num_orientations=num_orientations,
            clockwise=clockwise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        num_orientations = ctx.num_orientations
        clockwise = ctx.clockwise
        num_samples = ctx.num_samples
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, feature_h, feature_w = feature_size

        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, feature_h,
                                        feature_w)
            ext_module.riroi_align_rotated_backward(
                grad_output.contiguous(),
                rois,
                grad_input,
                pooled_height=out_h,
                pooled_width=out_w,
                spatial_scale=spatial_scale,
                num_samples=num_samples,
                num_orientations=num_orientations,
                clockwise=clockwise)

            return grad_input, grad_rois, None, None, None, None, None


riroi_align_rotated = RiRoIAlignRotatedFunction.apply


class RiRoIAlignRotated(nn.Module):
    """Rotation-invariant RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    The details are described in the paper `ReDet: A Rotation-equivariant
    Detector for Aerial Object Detection  <https://arxiv.org/abs/2103.07733>`_.

    Args:
        out_size (tuple): fixed dimensional RoI output with shape (h, w).
        spatial_scale (float): scale the input boxes by this number
        num_samples (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        num_orientations (int): number of oriented channels.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.
    """

    def __init__(self,
                 out_size,
                 spatial_scale,
                 num_samples=0,
                 num_orientations=8,
                 clockwise=False):
        super(RiRoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.num_samples = int(num_samples)
        self.num_orientations = int(num_orientations)
        self.clockwise = clockwise

    def forward(self, features, rois):
        return RiRoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                               self.spatial_scale,
                                               self.num_samples,
                                               self.num_orientations,
                                               self.clockwise)
