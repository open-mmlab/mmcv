# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
from typing import Dict, Optional

import torch
import yaml

import mmcv
from ....parallel.utils import is_module_wrapper
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class PaviLoggerHook(LoggerHook):
    """Class to visual model, log metrics (for internal use).

    Args:
        init_kwargs (dict): A dict contains the initialization keys as below:
            - name (str, optional): Custom training name. Defaults to None,
              which means current work_dir.
            - project (str, optional): Project name. Defaults to "default".
            - model (str, optional): Training model name. Defaults to current
              model.
            - session_text (str, optional): Session string in YAML format.
              Defaults to current config.
            - training_id (int, optional): Training ID in PAVI, if you want to
              use an existing training. Defaults to None.
            - compare_id (int, optional): Compare ID in PAVI, if you want to
              add the task to an existing compare. Defaults to None.
            - overwrite_last_training (bool, optional): Whether to upload data
              to the training with the same name in the same project, rather
              than creating a new one. Defaults to False.
        add_graph (bool): Whether to visual model. Default: False.
        add_graph_defined (dict, optional): Define the params for adding graph.
            Default: {'start_epoch': 0, 'interval_epoch': 1}.
        add_last_ckpt (bool): Whether to save checkpoint after run.
            Default: False.
        add_ckpt_defined (dict, optional): Define the params for adding
            checkpoint. Default: {'start_epoch': 0, 'interval_epoch': 1}.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        img_key (string): Get image data from Dataset. Default: 'img_info'.
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 add_graph: bool = False,
                 add_graph_defined: dict = {
                     'start_epoch': 0,
                     'interval_epoch': 1
                 },
                 add_last_ckpt: bool = False,
                 add_ckpt_defined: dict = {
                     'start_epoch': 0,
                     'interval_epoch': 1
                 },
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True,
                 img_key: str = 'img_info') -> None:
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.init_kwargs = init_kwargs
        self.add_graph = add_graph
        self.add_graph_start_epoch = add_graph_defined.get('start_epoch', 0)
        self.add_graph_interval_epoch = add_graph_defined.get(
            'interval_epoch', 1)
        self.add_last_ckpt = add_last_ckpt
        self.add_ckpt_start_epoch = add_ckpt_defined.get('start_epoch', 0)
        self.add_ckpt_interval_epoch = add_ckpt_defined.get(
            'interval_epoch', 0)
        self.img_key = img_key

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError('Please run "pip install pavi" to install pavi.')

        self.run_name = runner.work_dir.split('/')[-1]

        if not self.init_kwargs:
            self.init_kwargs = dict()
        self.init_kwargs.setdefault('name', self.run_name)
        self.init_kwargs.setdefault('model', runner._model_name)
        if runner.meta is not None:
            if 'config_dict' in runner.meta:
                config_dict = runner.meta['config_dict']
                assert isinstance(
                    config_dict,
                    dict), ('meta["config_dict"] has to be of a dict, '
                            f'but got {type(config_dict)}')
            elif 'config_file' in runner.meta:
                config_file = runner.meta['config_file']
                config_dict = dict(mmcv.Config.fromfile(config_file))
            else:
                config_dict = None
            if config_dict is not None:
                # 'max_.*iter' is parsed in pavi sdk as the maximum iterations
                #  to properly set up the progress bar.
                config_dict = config_dict.copy()
                config_dict.setdefault('max_iter', runner.max_iters)
                # non-serializable values are first converted in
                # mmcv.dump to json
                config_dict = json.loads(
                    mmcv.dump(config_dict, file_format='json'))
                session_text = yaml.dump(config_dict)
                self.init_kwargs.setdefault('session_text', session_text)
        self.writer = SummaryWriter(**self.init_kwargs)

    def get_step(self, runner) -> int:
        """Get the total training step/epoch."""
        if self.get_mode(runner) == 'val' and self.by_epoch:
            return self.get_epoch(runner)
        else:
            return self.get_iter(runner)

    def _add_ckpt(self, runner) -> None:
        ckpt_path = osp.join(runner.work_dir, 'latest.pth')
        if osp.islink(ckpt_path):
            ckpt_path = osp.join(runner.work_dir, os.readlink(ckpt_path))

        if osp.isfile(ckpt_path):
            # runner.epoch += 1 has been done before `after_run`.
            iteration = runner.epoch if self.by_epoch else runner.iter
            self.writer.add_snapshot_file(
                tag=self.run_name,
                snapshot_file_path=ckpt_path,
                iteration=iteration)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, add_mode=False)
        if tags:
            self.writer.add_scalars(
                self.get_mode(runner), tags, self.get_step(runner))

    @master_only
    def after_run(self, runner) -> None:
        if self.add_last_ckpt:
            self._add_ckpt(runner)

        # flush the buffer and send a task ending signal to Pavi
        self.writer.close()

    @master_only
    def before_epoch(self, runner) -> None:
        if self.add_graph and (runner.epoch % self.add_graph_interval_epoch
                               == self.add_graph_start_epoch):
            if is_module_wrapper(runner.model):
                _model = runner.model.module
            else:
                _model = runner.model
            device = next(_model.parameters()).device
            data = next(iter(runner.data_loader))
            image = data[self.img_key][0:1].to(device)
            with torch.no_grad():
                self.writer.add_graph(_model, image)

    @master_only
    def after_train_epoch(self, runner) -> None:
        if runner.epoch % self.add_ckpt_interval_epoch == \
                self.add_ckpt_start_epoch:
            self._add_ckpt(runner)
