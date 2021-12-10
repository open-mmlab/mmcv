# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['feature_refine_forward', 'feature_refine_backward'])


class FeatureRefineFunction(Function):
    """Feature refine class."""

    @staticmethod
    def forward(ctx, features, best_rbboxes, spatial_scale, points=1):
        """Forward function."""
        ctx.spatial_scale = spatial_scale
        ctx.points = points
        ctx.save_for_backward(best_rbboxes)
        assert points in [1, 5]
        assert features.is_cuda
        output = torch.zeros_like(features)
        ext_module.feature_refine_forward(features, best_rbboxes,
                                          spatial_scale, points, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Backward function."""
        best_rbboxes = ctx.saved_tensors[0]
        points = ctx.points
        spatial_scale = ctx.spatial_scale
        assert grad_output.is_cuda
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(grad_output)
            ext_module.feature_refine_backward(grad_output.contiguous(),
                                               best_rbboxes, spatial_scale,
                                               points, grad_input)
        return grad_input, None, None, None


feature_refine = FeatureRefineFunction.apply
