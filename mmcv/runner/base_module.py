# Copyright (c) Open-MMLab. All rights reserved.

from abc import ABCMeta

import torch.nn as nn

from mmcv.cnn import initialize


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab"""

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`

        Args:
            init_cfg (dict, optional): Initialization config dict.
        """

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weigt() function

        if init_cfg is not None:
            self.init_cfg = init_cfg

        # Backward compatibility
        # if pretrained is not None:
        #     self.init_cfg = dict(
        #         type='pretrained', checkpoint=pretrained)

    def init_weight(self):
        """Initialize the weights.
        """
        if hasattr(self, 'init_cfg'):
            initialize(self, self.init_cfg)
        for module in self.children():
            if 'init_weight' in dir(module):
                module.init_weight()
