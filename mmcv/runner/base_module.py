# Copyright (c) Open-MMLab. All rights reserved.
import warnings
from abc import ABCMeta
from collections import defaultdict

import torch.nn as nn

from mmcv.utils.logging import logger_initialized, print_log


def update_init_infos(module, *, init_info):
    """Update the `params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The `module of PyTorch with a user-define
            attributes `params_init_info` which recorde the initialization
            information.
        init_info (str): The string describes the initialization.
    """
    for param in module.parameters():
        if module.params_init_info[param]['tmp_sum_value'] != param.data.sum():
            module.params_init_info[param]['init_info'] = init_info
            module.params_init_info[param]['tmp_sum_value'] = param.data.sum()


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
        # in init_weight() function
        self._is_init = False
        self.init_cfg = init_cfg

        # The `params_init_info` is used to record the initialization
        # information of the parameters
        # the key should be the obj:`nn.Parameter` of model and the value
        # should be a dict contains
        # `params_name`, `init_info` and `tmp_sum_value`.
        # this attribute would be deleted after all parameters is initialized.
        self.params_init_info = defaultdict(dict)

        # Backward compatibility in derived classes
        # if pretrained is not None:
        #     warnings.warn('DeprecationWarning: pretrained is a deprecated \
        #         key, please consider using init_cfg')
        #     self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """Initialize the weights."""

        # judge if it is Topmost module
        top_most_module = len(self.params_init_info) == 0
        if top_most_module:
            for name, param in self.named_parameters():
                self.params_init_info[param]['params_name'] = name
                self.params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self.params_init_info[param]['tmp_sum_value'] = param.data.sum(
                )
            # pass `params_init_info` to all submodules
            # all submodules will modify the same `params_init_info` \
            # during initialization thus params_init_info will
            # keep the final initialization
            # information of each module.
            for sub_moduls in self.modules():
                sub_moduls.params_init_info = self.params_init_info

        loggernames = list(logger_initialized.keys())
        loggername = loggernames[0] if len(loggernames) > 0 else 'mmcv'

        from ..cnn import initialize
        modulename = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {modulename} with init_cfg {self.init_cfg}',
                    logger=loggername)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # Avoid the parameters of the pre-training model
                    # being overwritten by the init_weights
                    # of the children.
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # user may overload the `init_weights`
                    update_init_infos(
                        m,
                        init_info='Initialized by user-defined `init_weights`')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if top_most_module:
            for item in list(self.params_init_info.values()):
                print_log(
                    f"{item['params_name']} - {item['init_info']}",
                    logger=loggername)
            for sub_moduls in self.modules():
                del sub_moduls.params_init_info

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
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
