# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch.nn as nn
from mmengine.utils import deprecated_api_warning
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['roi_align_rotated_v2_forward'])


class RoIAlignRotatedV2Function(Function):

    @staticmethod
    def symbolic(g, input, rois, spatial_scale, sampling_ratio, pooled_height,
                 pooled_width, aligned, clockwise):
        return g.op(
            'mmcv::MMCVRoIAlignRotatedV2',
            input,
            rois,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            aligned_i=aligned,
            clockwise_i=clockwise)

    @staticmethod
    def forward(ctx: Any,
                input: torch.Tensor,
                rois: torch.Tensor,
                spatial_scale: float,
                sampling_ratio: int,
                pooled_height: int,
                pooled_width: int,
                aligned: bool = True,
                clockwise: bool = False) -> torch.Tensor:
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.aligned = aligned
        ctx.clockwise = clockwise
        ctx.save_for_backward(input, rois)
        ctx.feature_size = input.size()
        batch_size, num_channels, data_height, data_width = input.size()
        num_rois = rois.size(0)

        output = input.new_zeros(num_rois, ctx.pooled_height, ctx.pooled_width,
                                 num_channels)

        ext_module.roi_align_rotated_v2_forward(
            input,
            rois,
            output,
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pooled_height=ctx.pooled_height,
            pooled_width=ctx.pooled_width,
            aligned=ctx.aligned,
            clockwise=ctx.clockwise)
        output = output.transpose(2, 3).transpose(1, 2).contiguous()
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        input, rois = ctx.saved_tensors
        rois_trans = torch.permute(rois, (1, 0)).contiguous()
        grad_output_trans = torch.permute(grad_output,
                                          (0, 2, 3, 1)).contiguous()
        grad_input = input.new_zeros(
            input.size(0), input.size(2), input.size(3), input.size(1))
        ext_module.roi_align_rotated_v2_backward(
            input, rois_trans, grad_output_trans, grad_input,
            ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
            ctx.sampling_ratio, ctx.aligned, ctx.clockwise)
        grad_input = grad_input.permute(0, 3, 1, 2).contiguous()

        return grad_input, None, None, None, None, None, None, None


roi_align_rotated_v2 = RoIAlignRotatedV2Function.apply


class RoIAlignRotatedV2(nn.Module):
    """RoI align pooling layer for rotated proposals.

    It accepts a feature map of shape (N, C, H, W) and rois with shape
    (n, 6) with each roi decoded as (batch_index, center_x, center_y,
    w, h, angle). The angle is in radian.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio(int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
            Default: True.
        clockwise (bool): If True, the angle in each proposal follows a
            clockwise fashion in image space, otherwise, the angle is
            counterclockwise. Default: False.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """

    @deprecated_api_warning(
        {
            'out_size': 'output_size',
            'sample_num': 'sampling_ratio'
        },
        cls_name='RoIAlignRotatedV2')
    def __init__(self,
                 spatial_scale: float,
                 sampling_ratio: int,
                 pooled_height: int,
                 pooled_width: int,
                 aligned: bool = True,
                 clockwise: bool = False):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = aligned
        self.clockwise = clockwise

    def forward(self, input: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return RoIAlignRotatedV2Function.apply(input, rois, self.spatial_scale,
                                               self.sampling_ratio,
                                               self.pooled_height,
                                               self.pooled_width, self.aligned,
                                               self.clockwise)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(pooled_height={self.pooled_height}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'aligned={self.aligned}, '
        s += f'clockwise={self.clockwise})'
        return s
