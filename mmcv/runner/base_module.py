# Copyright (c) Open-MMLab. All rights reserved.
import copy
import warnings
from abc import ABCMeta
from collections import defaultdict

import torch.nn as nn

from mmcv.runner.dist_utils import master_only
from mmcv.utils.logging import get_logger, logger_initialized, print_log


def update_init_info(module, *, init_info):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    for param in module.parameters():
        mean_value = param.data.mean()
        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
            module._params_init_info[param]['init_info'] = init_info
            module._params_init_info[param]['tmp_mean_value'] = mean_value


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab.

    ``BaseModule`` is a wrapper of ``torch.nn.Module`` with additional
    functionality of parameters initialization. Compared with
    ``torch.nn.Module``, ``BaseModule`` mainly added two attributes
    ``init_cfg`` the config for initialization, ``_params_init_info`` to
    track the parameters initialization information and one function
    ``init_weigths`` to implement the functions of parameters
    initialization and initialization information record.

    Args:
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self, init_cfg=None):
        """Initialize BaseModule, inherited from `torch.nn.Module`"""

        # NOTE init_cfg can be defined in different levels, but init_cfg
        # in low levels has a higher priority.

        super(BaseModule, self).__init__()
        # define default value of init_cfg instead of hard code
        # in init_weight() function
        self._is_init = False
        if init_cfg:
            init_cfg = copy.deepcopy(init_cfg)
        self.init_cfg = init_cfg

        # The `_params_init_info` is used to record the initialization
        # information of the parameters
        # the key should be the obj:`nn.Parameter` of model and the value
        # should be a dict containing
        # - param_name (str): The name of parameter
        # - init_info (str): The string that describes the initialization.
        # - tmp_mean_value (FloatTensor): The mean of the parameter,
        #       which indicates whether the parameter has been modified.
        # this attribute would be deleted after all parameters is initialized.
        self._params_init_info = defaultdict(dict)

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

        # check if it is top-level module
        is_top_level_module = len(self._params_init_info) == 0
        if is_top_level_module:
            # Initialize the `_params_init_info`,
            # When detecting the `tmp_mean_value` of
            # the corresponding parameter is changed, update related
            # initialization information
            for name, param in self.named_parameters():
                self._params_init_info[param]['param_name'] = name
                self._params_init_info[param][
                    'init_info'] = f'The value is the same before and ' \
                                   f'after calling `init_weights` ' \
                                   f'of {self.__class__.__name__} '
                self._params_init_info[param][
                    'tmp_mean_value'] = param.data.mean()

            # pass `params_init_info` to all submodules
            # All submodules share the same `params_init_info`,
            # so it will be updated when parameters are
            # modified at any level of the model.
            for sub_module in self.modules():
                sub_module._params_init_info = self._params_init_info

        # Get the initialized logger, if not exist,
        # create a logger named `mmcv`
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else 'mmcv'

        from ..cnn import initialize
        module_name = self.__class__.__name__
        if not self._is_init:
            if self.init_cfg:
                print_log(
                    f'initialize {module_name} with init_cfg {self.init_cfg}',
                    logger=logger_name)
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, dict):
                    # Avoid the parameters of the pretrained model
                    # being overwritten by the `init_weights`
                    # of the children.
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
                    # users may overload the `init_weights`
                    update_init_info(
                        m,
                        init_info=f'Initialized by '
                        f'user-defined `init_weights`'
                        f' in {m.__class__.__name__} ')

            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has '
                          f'been called more than once.')

        if is_top_level_module:
            self._dump_init_info(logger_name)

            for sub_module in self.modules():
                del sub_module._params_init_info

    @master_only
    def _dump_init_info(self, logger_name):
        """Dump the initialization information to a file named
        `initialization.log.json` in workdir.

        Args:
            logger_name (str): The name of logger.
        """
        logger = get_logger(logger_name)
        logger_file = None

        # get workdir from file_handler
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                logger_file = handler.baseFilename

        # if can get workdir from `file_handler`, write
        # initialization information to a file named
        # {time_prefix}_initialization.log in workdir.
        # otherwise just print it
        if logger_file:
            logger_file_name = logger_file.split('/')[-1]
            # %Y%m%d_%H%M%S
            time_prefix = logger_file_name.split('.')[0]

            init_logger_file = logger_file.replace(
                logger_file_name, f'{time_prefix}_initialization.log')

            with open(init_logger_file, 'w') as f:
                f.write('Name of parameter - Initialization information\n')
                for item in list(self._params_init_info.values()):
                    f.write(f"{item['param_name']} - {item['init_info']} \n")
        else:
            for item in list(self._params_init_info.values()):
                print_log(
                    f"{item['param_name']} - {item['init_info']}",
                    logger=logger_name)

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
