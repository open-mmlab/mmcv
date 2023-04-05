# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmcv.runner import HOOKS, Hook
from .utils import replace_bn


@HOOKS.register_module()
class BNToFP32(Hook):

    def bn2float(self, module):
        for _name, _module in module.named_children():
            if isinstance(_module, torch.nn.BatchNorm2d):
                _module.float()
            else:
                self.bn2float(_module)

    def before_run(self, runner):
        self.bn2float(runner.model)


@HOOKS.register_module()
class BNToIPUBN(Hook):

    def before_run(self, runner):
        replace_bn(runner.model)
