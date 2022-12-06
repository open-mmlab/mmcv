# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['bezier_align_forward', 'bezier_align_backward'])


class BezierAlignFunction(Function):

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                beziers: torch.Tensor,
                output_size: Union[int, Tuple[int, int]],
                spatial_scale: Union[int, float] = 1.0,
                sampling_ratio: int = 0,
                aligned: bool = True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        ctx.sampling_ratio = sampling_ratio
        ctx.aligned = aligned

        assert beziers.size(1) == 17
        output_shape = (beziers.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        ext_module.bezier_align_forward(
            input,
            beziers,
            output,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            aligned=ctx.aligned)

        ctx.save_for_backward(beziers)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        beziers = ctx.saved_tensors[0]
        grad_input = grad_output.new_zeros(ctx.input_shape)
        grad_output = grad_output.contiguous()
        ext_module.bezier_align_backward(
            grad_output,
            beziers,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None


bezier_align = BezierAlignFunction.apply


class BezierAlign(nn.Module):
    """Bezier align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.

    Note:
        The implementation of BezierAlign is modified from
        https://github.com/aim-uofa/AdelaiDet

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

    def __init__(
        self,
        output_size,
        spatial_scale,
        sampling_ratio,
        aligned: bool = True,
    ):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.aligned = aligned

    def forward(self, input, beziers):
        """"""
        return bezier_align(input, beziers, self.output_size,
                            self.spatial_scale, self.sampling_ratio,
                            self.aligned)

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale})'
        s += f'sampling_ratio={self.sampling_ratio})'
        s += f'aligned={self.aligned})'
        return s
