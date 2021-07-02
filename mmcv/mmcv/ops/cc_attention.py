import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from mmcv.cnn import Scale
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['ca_forward', 'ca_backward', 'ca_map_forward', 'ca_map_backward'])


class CAWeightFunction(torch.autograd.Function):

    @staticmethod
    def symbolic(g, t, f):
        return g.op('MMCVCAWeight', t, f)

    @staticmethod
    def forward(ctx, t, f):
        n, c, h, w = t.size()
        weight = torch.zeros(n, h + w - 1, h, w).to(t.device)
        ext_module.ca_forward(t, f, weight)

        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors
        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)
        ext_module.ca_backward(dw, t, f, dt, df)
        return dt, df


class CAMapFunction(torch.autograd.Function):

    @staticmethod
    def symbolic(g, weight, v):
        return g.op('MMCVCAMap', weight, v)

    @staticmethod
    def forward(ctx, weight, v):
        out = torch.zeros_like(v)
        ext_module.ca_map_forward(weight, v, out)

        ctx.save_for_backward(weight, v)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, v = ctx.saved_tensors
        dw = torch.zeros_like(weight)
        dv = torch.zeros_like(v)
        ext_module.ca_map_backward(dout, weight, v, dw, dv)

        return dw, dv


ca_weight = CAWeightFunction.apply
ca_map = CAMapFunction.apply


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module."""

    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = Scale(0.)
        self.in_channels = in_channels

    def forward(self, x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma(out) + x

        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels})'
        return s
