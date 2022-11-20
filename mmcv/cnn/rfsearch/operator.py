# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .utils import expands_rate, get_single_padding
from mmcv.runner import BaseModule
from mmcv.utils.logging import get_logger

logger = get_logger('Operators')


class ConvRFSearchOp(BaseModule):
    """Based class of ConvRFSearchOp.

    Args:
        op_layer (nn.Module): pytorch module, e,g, Conv2d
        global_config (Dict): config dict.
    """

    def __init__(self, op_layer: nn.Module, global_config: dict):
        super().__init__()
        self.op_layer = op_layer
        self.global_config = global_config

    def normlize(self, weights: nn.Parameter) -> nn.Parameter:
        """Normalize weights.

        Args:
            weights (nn.Parameter): Weights to be normalized.

        Returns:
            nn.Parameters: Normalized weights.
        """
        abs_weights = torch.abs(weights)
        normalized_weights = abs_weights / torch.sum(abs_weights)
        return normalized_weights


class Conv2dRFSearchOp(ConvRFSearchOp):
    """Enable Conv2d with receptive field searching ability.

    Args:
        op_layer (nn.Module): pytorch module, e,g, Conv2d
        init_dilation (list, optional): Initial dilation rate. Defaults to None.
        global_config (dict): config dict. Defaults to None.
            By default this must include:

            - "init_alphas": The value for initializing weights of each branch.
            - "num_branches": The controller of the size of
              search space (the number of branches).
            - "exp_rate": The controller of the sparsity of search space.
            - "mmin": The minimum dilation rate.
            - "mmax": The maximum dilation rate.

            Extra keys may exist, but are used by RFSearchHook, e.g., "step",
            "max_step", "search_interval", and "skip_layer".
        num_branches (int, optional): Number of branches. Defaults to 3.
        verbose (bool): Determines whether to print rf-next
            related logging messages.
            Defaults to True.
    """

    def __init__(self,
                 op_layer: nn.Module,
                 init_dilation: list = None,
                 global_config: Optional[Dict] = None,
                 verbose: bool = True):
        super().__init__(op_layer, global_config)
        assert global_config is not None, 'global_config is None'
        self.num_branches = global_config['num_branches']
        assert self.num_branches in [2, 3]
        self.verbose = verbose
        if init_dilation is None:
            init_dilation = op_layer.dilation
        self.rates = expands_rate(init_dilation, global_config)
        if self.op_layer.kernel_size[
                0] == 1 or self.op_layer.kernel_size[0] % 2 == 0:
            self.rates = [(op_layer.dilation[0], r[1]) for r in self.rates]
        if self.op_layer.kernel_size[
                1] == 1 or self.op_layer.kernel_size[1] % 2 == 0:
            self.rates = [(r[0], op_layer.dilation[1]) for r in self.rates]

        self.weights = nn.Parameter(torch.Tensor(self.num_branches))
        if self.verbose:
            logger.info(f'Expand as {self.rates}')
        nn.init.constant_(self.weights, global_config['init_alphas'])

    def forward(self, x: Tensor) -> Tensor:
        norm_w = self.normlize(self.weights[:len(self.rates)])
        if len(self.rates) == 1:
            xx = [
                nn.functional.conv2d(
                    x,
                    weight=self.op_layer.weight,
                    bias=self.op_layer.bias,
                    stride=self.op_layer.stride,
                    padding=self.get_padding(self.rates[0]),
                    dilation=self.rates[0],
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
                    padding=self.get_padding(r),
                    dilation=r,
                    groups=self.op_layer.groups,
                ) * norm_w[i] for i, r in enumerate(self.rates)
            ]
        x = xx[0]
        for i in range(1, len(self.rates)):
            x += xx[i]
        return x

    def estimate(self):
        """estimate new dilation rate based on trained weights."""
        norm_w = self.normlize(self.weights[:len(self.rates)])
        if self.verbose:
            logger.info('Estimate dilation {} with weight {}.'.format(
                self.rates,
                norm_w.detach().cpu().numpy().tolist()))

        sum0, sum1, w_sum = 0, 0, 0
        for i in range(len(self.rates)):
            sum0 += norm_w[i].item() * self.rates[i][0]
            sum1 += norm_w[i].item() * self.rates[i][1]
            w_sum += norm_w[i].item()
        estimated = [
            np.clip(
                int(round(sum0 / w_sum)), self.global_config['mmin'],
                self.global_config['mmax']).item(),
            np.clip(
                int(round(sum1 / w_sum)), self.global_config['mmin'],
                self.global_config['mmax']).item()
        ]
        self.op_layer.dilation = tuple(estimated)
        self.op_layer.padding = self.get_padding(self.op_layer.dilation)
        self.rates = [tuple(estimated)]
        if self.verbose:
            logger.info(f'Estimate as {tuple(estimated)}')

    def expand(self):
        """expand dilation rate."""
        d = self.op_layer.dilation
        rates = expands_rate(d, self.global_config)
        if self.op_layer.kernel_size[
                0] == 1 or self.op_layer.kernel_size[0] % 2 == 0:
            rates = [(d[0], r[1]) for r in rates]
        if self.op_layer.kernel_size[
                1] == 1 or self.op_layer.kernel_size[1] % 2 == 0:
            rates = [(r[0], d[1]) for r in rates]

        self.rates = copy.deepcopy(rates)
        if self.verbose:
            logger.info(f'Expand as {self.rates}')
        nn.init.constant_(self.weights, self.global_config['init_alphas'])

    def get_padding(self, dilation):
        padding = (get_single_padding(self.op_layer.kernel_size[0],
                                      self.op_layer.stride[0], dilation[0]),
                   get_single_padding(self.op_layer.kernel_size[1],
                                      self.op_layer.stride[1], dilation[1]))
        return padding
