# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from abc import ABCMeta, abstractmethod
# import poptorch

import mmcv
from .iter_based_runner import IterBasedRunner
from .epoch_based_runner import EpochBasedRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .utils import get_host_info


def parse_ipu_options(ipu_options):
    return ipu_options


def wrap_model(model, opts, optimizer):
    # model = poptorch.trainingModel(model, options=opts, optimizer=optimizer)
    return model


def wrap_data_loader(data_loader, opts):
    pass
    return data_loader


class IpuBaseRunner(metaclass=ABCMeta):
    def __init__(self,ipu_options=None,**kwargs):
        super(IpuBaseRunner, self).__init__(**kwargs)
        # process options of ipu
        self.ipu_options = parse_ipu_options(ipu_options)
        # self.data_loader = wrap_data_loader(self.data_loader)
        self.model = wrap_model(self.model, self.ipu_options, self.optimizer)
        self.ipu_data_loaders_mappin = {}


    def run(self, data_loaders, *args, **kwargs):
        # map data_loader to ipu data_loader
        ipu_data_loaders = []
        for data_loader in data_loaders:
            if data_loader not in self.ipu_data_loaders_mappin:
                ipu_data_loader = wrap_data_loader(data_loader)
                self.ipu_data_loaders_mappin[data_loader] = ipu_data_loader
            else:
                ipu_data_loader = self.ipu_data_loaders_mappin[data_loader]
            ipu_data_loaders.append(ipu_data_loader)
        super().run(ipu_data_loaders, *args, **kwargs)


@RUNNERS.register_module()
class IpuEpochBasedRunner(IpuBaseRunner, EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    pass


@RUNNERS.register_module()
class IpuIterBasedRunner(IpuBaseRunner, IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """
    pass

