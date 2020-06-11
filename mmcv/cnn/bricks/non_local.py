from abc import ABCMeta

import torch
import torch.nn as nn

from ..weight_init import constant_init, normal_init
from .conv_module import ConvModule


class _NonLocalNd(nn.Module, metaclass=ABCMeta):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
            Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 **kwargs):
        super(_NonLocalNd, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode

        if mode not in ['embedded_gaussian', 'dot_product']:
            raise ValueError(
                "Mode should be in 'embedded_gaussian' or 'dot_product', "
                f'but got {mode} instead.')

        # g, theta, phi are defaulted as `nn.Conv2d`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.init_weights(**kwargs)

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
            if self.conv_out.norm_cfg is not None:
                constant_init(self.conv_out.norm, 0)
        else:
            normal_init(self.conv_out.conv, std=std)
            if self.conv_out.norm_cfg is not None:
                normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # Here we take NonLocal2D tensor shape change for example
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    @staticmethod
    def dot_product(theta_x, phi_x):
        # Here we take NonLocal2D tensor shape change for example
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        # Here we take NonLocal2D tensor shape change for example
        # n, _, h, w = x.shape
        n = x.size(0)

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, *x.size()[2:])

        output = x + self.conv_out(y)

        return output


class NonLocal1d(_NonLocalNd):
    """1D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function. Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv1d').
    """

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv1d'),
                 **kwargs):
        super(NonLocal1d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample
        max_pool_layer = nn.MaxPool1d(kernel_size=2)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function. Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv2d').
    """

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv2d'),
                 **kwargs):
        super(NonLocal2d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


class NonLocal3d(_NonLocalNd):
    """3D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function. Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='Conv3d').
    """

    def __init__(self,
                 in_channels,
                 sub_sample=False,
                 conv_cfg=dict(type='Conv3d'),
                 **kwargs):
        super(NonLocal3d, self).__init__(
            in_channels, conv_cfg=conv_cfg, **kwargs)
        self.sub_sample = sub_sample
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
