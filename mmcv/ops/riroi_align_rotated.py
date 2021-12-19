# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['riroi_align_rotated_forward', 'riroi_align_rotated_backward'])


class RiRoIAlignRotatedFunction(Function):

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                sample_num=0,
                nOrientation=8,
                clockwise=False):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.nOrientation = nOrientation
        ctx.clockwise = clockwise
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)

        ext_module.riroi_align_rotated_forward(features, rois, output, out_h,
                                               out_w, spatial_scale,
                                               sample_num, nOrientation,
                                               clockwise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        nOrientation = ctx.nOrientation
        clockwise = ctx.clockwise
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size

        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            ext_module.riroi_align_rotated_backward(grad_output.contiguous(),
                                                    rois, grad_input, out_h,
                                                    out_w, spatial_scale,
                                                    sample_num, nOrientation,
                                                    clockwise)

            return grad_input, grad_rois, None, None, None, None, None


riroi_align_rotated = RiRoIAlignRotatedFunction.apply


class RiRoIAlignRotated(nn.Module):
    """RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    Args:
        out_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sample_num (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        nOrientation (int): number of oriented channels.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.
    """

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 nOrientation=8,
                 clockwise=False):
        super(RiRoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.nOrientation = int(nOrientation)
        self.clockwise = clockwise

    def forward(self, features, rois):
        return RiRoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                               self.spatial_scale,
                                               self.sample_num,
                                               self.nOrientation,
                                               self.clockwise)
