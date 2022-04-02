import torch
from torch import nn
from torch.autograd import Function
from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'diff_iou_rotated_sort_vertices'
])

class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        idx = ext_module.diff_iou_rotated_sort_vertices(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()

diff_iou_rotated_sort_vertices = SortVertices.apply

