# Copyright (c) Open-MMLab. All rights reserved.
import warnings
from abc import ABCMeta

import torch.nn as nn


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab."""

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
        self._is_init = False
        if init_cfg is not None:
            self.init_cfg = init_cfg

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weight(self):
        """Initialize the weights."""
        from ..cnn import initialize

        if not self._is_init:
            if hasattr(self, 'init_cfg'):
                initialize(self, self.init_cfg)
            for m in self.children():
                if hasattr(m, 'init_weight'):
                    m.init_weight()
            self._is_init = True
        else:
            warnings.warn(f'init_weight of {self.__class__.__name__} has '
                          f'been called more than once.')

    def __repr__(self):
        s = super().__repr__()
        if hasattr(self, 'init_cfg'):
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class Sequential(BaseModule, nn.Sequential):
    """Sequential module in openmmlab.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """ModuleList in openmmlab.

    Args:
        modules (iterable, optional): an iterable of modules to add.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)
