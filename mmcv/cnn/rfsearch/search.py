# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from typing import Dict

import torch  # noqa
import torch.nn as nn

import mmcv
from mmcv.runner import HOOKS, Hook
from .operator import Conv2dRFSearchOp, ConvRFSearchOp  # noqa
from .utils import write_to_json

logging.basicConfig(
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.NOTSET,
    datefmt='%Y-%m-%d%I:%M:%S %p',
)
logger = logging.getLogger('Searcher')
logger.setLevel(logging.ERROR)


@HOOKS.register_module()
class RFSearchHook(Hook):
    """Rcecptive field search via dilation rates.
        Paper: RF-Next: Efficient Receptive Field
            Search for Convolutional Neural Networks

    Args:
        mode (str, optional):
                search/fixed_single_branch/fixed_multi_branch.
        config (Dict, optional): config dict of search.
        rfstructure_file (Optional[str], optional):
                searched recptive fields of the model.
    """

    def __init__(self,
                 mode: str = 'search',
                 config: Dict = {},
                 rfstructure_file: str = None):
        assert mode in ['search', 'fixed_single_branch', 'fixed_multi_branch']
        assert config is not None
        self.config = config
        self.config['structure'] = {}
        if rfstructure_file is not None:
            rfstructure = mmcv.load(rfstructure_file)['model']
            self.config['structure'] = rfstructure
        self.mode = mode
        self.S = self.config['search']['S']

    def model_init(self, model: nn.Module):
        """init model with search ability.

        Args:
            model (nn.Module): pytorch model

        Raises:
            NotImplementedError:
                only support three modes:
                    search/fixed_single_branch/fixed_multi_branch
        """
        print('RFSearch init begin.')
        if self.mode == 'search':
            if self.config['structure']:
                self.set_model(model, self.config, search_op='Conv2d')
            self.wrap_model(model, self.config, search_op='Conv2d')
        elif self.mode == 'fixed_single_branch':
            self.set_model(model, self.config, search_op='Conv2d')
        elif self.mode == 'fixed_multi_branch':
            self.set_model(model, self.config, search_op='Conv2d')
            self.wrap_model(model, self.config, search_op='Conv2d')
        else:
            raise NotImplementedError
        print('RFSearch init end.')
        pass

    def after_epoch(self, runner):
        """Do search after one training epoch.

        Args:
            runner (_type_): MMCV runner
        """
        if self.mode == 'search':
            print('Local-Search step begin.')
            self.step(runner.model, runner.work_dir)
            print('Local-Search step end.')
        pass

    def step(self, model: nn.Module, work_dir: str):
        """do one step of dilation search.

        Args:
            model (nn.Module): pytorch model
            work_dir (str): save path
        """
        self.config['search']['step'] += 1
        if (self.config['search']['step']
            ) % self.config['search']['search_interval'] == 0 and (self.config[
                'search']['step']) < self.config['search']['max_step']:
            self.search(model)
            for name, module in model.named_modules():
                if isinstance(module, ConvRFSearchOp):
                    self.config['structure'][name] = module.op_layer.dilation
            write_to_json(
                self.config,
                os.path.join(
                    work_dir,
                    'local_search_config_step%d.json' %
                    self.config['search']['step'],
                ),
            )
        elif (self.config['search']['step'] +
              1) == self.config['search']['max_step']:
            self.search_estimate_only(model)

    def search(self, model: nn.Module):
        """estimate and search for RFConvOp.

        Args:
            model (nn.Module): pytorch model
        """
        for _, module in model.named_modules():
            if isinstance(module, ConvRFSearchOp):
                module.estimate()
                module.expand()

    def search_estimate_only(self, model):
        for module in model.modules():
            if isinstance(module, ConvRFSearchOp):
                module.estimate()

    def wrap_model(self,
                   model: nn.Module,
                   config: Dict,
                   search_op: str = 'Conv2d',
                   init_rates: int = None):
        """wrap model to support searchable conv op.

        Args:
            model (nn.Module): pytorch model
            config (Dict): search config file
            search_op (str, optional):
                the module that uses RF search. Defaults to 'Conv2d'.
            init_rates (int, optional):
                Set to other initial dilation rates. Defaults to None.
        """
        op = 'torch.nn.' + search_op
        for name, module in model.named_children():
            if isinstance(module, eval(op)):
                if (1 < module.kernel_size[0]
                        and 0 != module.kernel_size[0] % 2):
                    moduleWrap = eval(search_op + 'RFSearchOp')(
                        module, init_rates, config['search'], self.S)
                    moduleWrap = moduleWrap.cuda()
                    logger.info('Wrap model %s to %s.' %
                                (str(module), str(moduleWrap)))
                    print('Wrap model %s to %s.' %
                          (str(module), str(moduleWrap)))
                    setattr(model, name, moduleWrap)
            elif isinstance(module, ConvRFSearchOp):
                pass
            else:
                if self.config['search']['skip_layer'] is not None:
                    if any([
                            each in name
                            for each in self.config['search']['skip_layer']
                    ]):
                        continue
                self.wrap_model(module, config, search_op, init_rates)

    def set_model(self,
                  model: nn.Module,
                  config: Dict,
                  search_op: str = 'Conv2d',
                  init_rates: int = None,
                  prefix: str = ''):
        """set model based on config.

        Args:
            model (nn.Module): pytorch model
            config (Dict): config file
            search_op (str, optional):
                the module that uses RF search. Defaults to 'Conv2d'.
            init_rates (int, optional):
                Set to other initial dilation rates. Defaults to None.
            prefix (str, optional):
                prefix for function recursion. Defaults to ''.
        """
        op = 'torch.nn.' + search_op
        for name, module in model.named_children():
            if prefix == '':
                fullname = 'module.' + name
            else:
                fullname = prefix + '.' + name
            if isinstance(module, eval(op)):
                if 1 < module.kernel_size[0] and \
                     0 != module.kernel_size[0] % 2:
                    if isinstance(config['structure'][fullname], int):
                        config['structure'][fullname] = [
                            config['structure'][fullname]
                        ]
                    module.dilation = (
                        config['structure'][fullname][0],
                        config['structure'][fullname][0],
                    )
                    module.padding = (
                        config['structure'][fullname][0] *
                        (module.kernel_size[0] - 1) // 2,
                        config['structure'][fullname][0] *
                        (module.kernel_size[0] - 1) // 2,
                    )
                    setattr(model, name, module)
                    logger.info('Set module %s dilation as: [%d]' %
                                (fullname, module.dilation[0]))
            elif isinstance(module, ConvRFSearchOp):
                pass
            else:
                if self.config['search']['skip_layer'] is not None:
                    if any([
                            each in fullname
                            for each in self.config['search']['skip_layer']
                    ]):
                        continue
                self.set_model(module, config, search_op, init_rates, fullname)
