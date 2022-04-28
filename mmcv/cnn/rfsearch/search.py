# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os

import torch  # noqa

from mmcv.runner import HOOKS, Hook
from .operator import BaseRFSearchOperator, Conv2dRFSearchOp  # noqa
from .utils import load_structure, write_to_json

logging.basicConfig(
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.NOTSET,
    datefmt='%Y-%m-%d%I:%M:%S %p',
)
logger = logging.getLogger('Searcher')
logger.setLevel(logging.ERROR)


@HOOKS.register_module()
class RFSearch(Hook):
    """Rcecptive field search via dilation rates.

    Paper: Efficient Receptive Field Search for Convolutional Neural Networks
    Args:
        logdir : save path of searched structure.
        mode : search/fixed_single_branch/fixed_multi_branch.
        config : config file of search.
        rfstructure_file : searched recptive fields of the model.
    """

    def __init__(self,
                 logdir='./log',
                 mode='search',
                 config=None,
                 rfstructure_file=None):
        assert logdir is not None
        assert mode in ['search', 'fixed_single_branch', 'fixed_multi_branch']
        assert config is not None
        self.config = config
        self.config['structure'] = {}
        if rfstructure_file is not None:
            rfstructure = load_structure(rfstructure_file)
            self.config['structure'] = rfstructure
        self.logdir = logdir
        self.mode = mode
        self.S = self.config['search']['S']
        os.makedirs(self.logdir, exist_ok=True)

    def model_init(self, model):
        print('RFSearch init begin.')
        # print(runner.model)
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

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        if self.mode == 'search':
            print('Local-Search step begin.')
            self.step(runner.model)
            print('Local-Search step end.')
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def step(self, model):
        self.config['search']['step'] += 1
        if (self.config['search']['step']
            ) % self.config['search']['search_interval'] == 0 and (self.config[
                'search']['step']) < self.config['search']['max_step']:
            self.search(model)
            for name, module in model.named_modules():
                if isinstance(module, BaseRFSearchOperator):
                    self.config['structure'][name] = module.op_layer.dilation
            write_to_json(
                self.config,
                os.path.join(
                    self.logdir,
                    'local_search_config_step%d.json' %
                    self.config['search']['step'],
                ),
            )
        elif (self.config['search']['step'] +
              1) == self.config['search']['max_step']:
            self.search_estimate_only(model)

    def search(self, model):
        for _, module in model.named_modules():
            if isinstance(module, BaseRFSearchOperator):
                module.estimate()
                module.expand()

    def search_estimate_only(self, model):
        for _, module in model.named_modules():
            if isinstance(module, BaseRFSearchOperator):
                module.estimate()

    def wrap_model(self, model, config, search_op='Conv2d', init_rates=None):
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
            elif isinstance(module, BaseRFSearchOperator):
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
                  model,
                  config,
                  search_op='Conv2d',
                  init_rates=None,
                  prefix=''):
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
            elif isinstance(module, BaseRFSearchOperator):
                pass
            else:
                if self.config['search']['skip_layer'] is not None:
                    if any([
                            each in fullname
                            for each in self.config['search']['skip_layer']
                    ]):
                        continue
                self.set_model(module, config, search_op, init_rates, fullname)
