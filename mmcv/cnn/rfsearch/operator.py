# Copyright (c) OpenMMLab. All rights reserved.
import abc
import copy
import logging

import torch
import torch.nn as nn

from .utils import expands_rate, value_crop

logger = logging.getLogger('Operators')
logger.setLevel(logging.INFO)


class BaseRFSearchOperator(abc.ABC):

    @abc.abstractmethod
    def estimate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def expand(self):
        raise NotImplementedError


class ConvRFSearchOp(BaseRFSearchOperator, nn.Module):

    def __init__(self, op_layer, global_config):
        super(ConvRFSearchOp, self).__init__()
        self.op_layer = op_layer
        self.global_config = global_config

    def normlize(self, w):
        if self.global_config['normlize'] == 'absavg':
            abs_w = torch.abs(w)
            norm_w = abs_w / torch.sum(abs_w)
        else:
            raise NotImplementedError
        return norm_w


class Conv2dRFSearchOp(ConvRFSearchOp):

    def __init__(self, op_layer, init_dilation, global_config, S=3):
        super().__init__(op_layer, global_config)
        assert S in [2, 3]
        self.S = S
        if init_dilation is None:
            init_dilation = op_layer.dilation[0]
        self.rates = expands_rate(init_dilation, global_config)
        self.weights = nn.Parameter(torch.Tensor(len(self.rates)))
        # When dilations is [1 1 x], change to double branch as [1 x]
        if init_dilation == 1:
            self.rates = self.rates[1:]
            logger.info('Expand to dilation %d %d' %
                        (self.rates[0], self.rates[1]))
        else:
            if self.S == 2:
                self.rates = [self.rates[0], self.rates[2]]
                logger.info('Expand to dilation %d %d' %
                            (self.rates[0], self.rates[1]))
            else:
                logger.info('Expand to dilation %d %d %d' %
                            (self.rates[0], self.rates[1], self.rates[2]))
        nn.init.constant_(self.weights, global_config['init_alphas'])

    def forward(self, x):
        norm_w = self.normlize(self.weights[:len(self.rates)])
        if len(self.rates) == 1:
            xx = [
                nn.functional.conv2d(
                    x,
                    weight=self.op_layer.weight,
                    bias=self.op_layer.bias,
                    stride=self.op_layer.stride,
                    padding=self.rates[0] *
                    (self.op_layer.kernel_size[0] - 1) // 2,
                    dilation=(self.rates[0], self.rates[0]),
                    groups=self.op_layer.groups,
                )
            ]
        else:
            xx = [
                nn.functional.conv2d(
                    x,
                    weight=self.op_layer.weight,
                    bias=self.op_layer.bias,
                    stride=self.op_layer.stride,
                    padding=r * (self.op_layer.kernel_size[0] - 1) // 2,
                    dilation=(r, r),
                    groups=self.op_layer.groups,
                ) * norm_w[i] for i, r in enumerate(self.rates)
            ]
        x = xx[0]
        for i in range(1, len(self.rates)):
            x += xx[i]
        return x

    def estimate(self):
        norm_w = self.normlize(self.weights[:len(self.rates)])
        if len(self.rates) == 2:
            logger.info('Estimate dilation %d %d with weight %f %f.' %
                        (self.rates[0], self.rates[1], norm_w[0].item(),
                         norm_w[1].item()))
        else:
            logger.info('Estimate dilation %d %d %d with weight %f %f %f.' % (
                self.rates[0],
                self.rates[1],
                self.rates[2],
                norm_w[0].item(),
                norm_w[1].item(),
                norm_w[2].item(),
            ))

        group_sum, w_sum = 0, 0
        for i in range(len(self.rates)):
            group_sum += norm_w[i].item() * self.rates[i]
            w_sum += norm_w[i].item()
        estimated = value_crop(
            int(round(group_sum / w_sum)),
            self.global_config['mmin'],
            self.global_config['mmax'],
        )
        self.op_layer.dilation = estimated
        self.rates = [estimated]
        logger.info('Estimate as %d' % estimated)

    def expand(self):
        d = self.op_layer.dilation
        rates = expands_rate(d, self.global_config)
        self.rates = copy.deepcopy(rates)
        # When dilations is [1 1 x], change to double branch as [1 x]
        if d == 1:
            self.rates = self.rates[1:]
            logger.info('Expand to dilation %d %d' %
                        (self.rates[0], self.rates[1]))
        else:
            if self.S == 2:
                self.rates = [self.rates[0], self.rates[2]]
                logger.info('Expand to dilation %d %d' %
                            (self.rates[0], self.rates[1]))
            else:
                logger.info('Expand to dilation %d %d %d' %
                            (self.rates[0], self.rates[1], self.rates[2]))
        nn.init.constant_(self.weights, self.global_config['init_alphas'])
