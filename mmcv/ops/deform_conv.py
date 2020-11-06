import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext('_ext', [
    'deform_conv_forward', 'deform_conv_backward_input',
    'deform_conv_backward_parameters'
])


class DeformConv2dFunction(Function):

    @staticmethod
    def symbolic(g,
                 input,
                 offset,
                 weight,
                 stride,
                 padding,
                 dilation,
                 groups,
                 deform_groups,
                 bias=False,
                 im2col_step=32):
        return g.op(
            'MMCVDeformConv2d',
            input,
            offset,
            weight,
            stride_i=stride,
            padding_i=padding,
            dilation_i=dilation,
            groups_i=groups,
            deform_groups_i=deform_groups,
            bias_i=bias,
            im2col_step_i=im2col_step)

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deform_groups=1,
                bias=False,
                im2col_step=32):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        assert bias is False, 'Only support bias is False.'
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.im2col_step = im2col_step

        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConv2dFunction._output_size(ctx, input, weight))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'
        ext_module.deform_conv_forward(
            input,
            weight,
            offset,
            output,
            ctx.bufs_[0],
            ctx.bufs_[1],
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[1],
            padH=ctx.padding[0],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            deformable_group=ctx.deform_groups,
            im2col_step=cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) %
                cur_im2col_step) == 0, 'im2col step must divide batchsize'

        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_offset = torch.zeros_like(offset)
            ext_module.deform_conv_backward_input(
                input,
                offset,
                grad_output,
                grad_input,
                grad_offset,
                weight,
                ctx.bufs_[0],
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                im2col_step=cur_im2col_step)

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            ext_module.deform_conv_backward_parameters(
                input,
                offset,
                grad_output,
                grad_weight,
                ctx.bufs_[0],
                ctx.bufs_[1],
                kW=weight.size(3),
                kH=weight.size(2),
                dW=ctx.stride[1],
                dH=ctx.stride[0],
                padW=ctx.padding[1],
                padH=ctx.padding[0],
                dilationW=ctx.dilation[1],
                dilationH=ctx.dilation[0],
                group=ctx.groups,
                deformable_group=ctx.deform_groups,
                scale=1,
                im2col_step=cur_im2col_step)

        return grad_input, grad_offset, grad_weight, \
            None, None, None, None, None, None, None

    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


deform_conv2d = DeformConv2dFunction.apply


class DeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='DeformConv2d')
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=False):
        super(DeformConv2d, self).__init__()

        assert not bias, \
            f'bias={bias} is not supported in DeformConv2d.'
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        # only weight, no bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out


@CONV_LAYERS.register_module('DCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super(DeformConv2dPack, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
