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
        add_graph (bool): **Deprecated**. Whether to visual model.
            Default: False.
        add_last_ckpt (bool): Whether to save checkpoint after run.
            Default: False.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        img_key (string): Get image data from Dataset. Default: 'img_info'.
        add_graph_kwargs (dict, optional): A dict contains the params for
            adding graph, the keys are as below:
            Default: {'active': False, 'start': 0, 'interval': 1}.
        add_ckpt_kwargs (dict, optional): A dict contains the params for
            adding checkpoint, the keys are as below:
            Default: {'active': False, 'start': 0, 'interval': 1}.
    """

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 add_graph: bool = False,
                 add_last_ckpt: bool = False,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True,
                 img_key: str = 'img_info',
                 add_graph_kwargs: Optional[Dict] = None,
                 add_ckpt_kwargs: Optional[Dict] = None) -> None:
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.init_kwargs = init_kwargs
        add_graph_kwargs = {} if add_graph_kwargs is None else add_graph_kwargs
        self.add_graph = add_graph_kwargs.get('active', False)
        self.add_graph_start = add_graph_kwargs.get('start', 0)
        self.add_graph_interval = add_graph_kwargs.get('interval', 1)

        add_ckpt_kwargs = {} if add_ckpt_kwargs is None else add_ckpt_kwargs
        self.add_ckpt = add_ckpt_kwargs.get('active', False)
        self.add_last_ckpt = add_last_ckpt
        self.add_ckpt_start = add_ckpt_kwargs.get('start', 0)
        self.add_ckpt_interval = add_ckpt_kwargs.get('interval', 1)
        self.img_key = img_key

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        try:
            from pavi import SummaryWriter
        except ImportError:
            raise ImportError(
                'No module named pavi, please contact pavi team or visit'
                'document for pavi installation instructions.')

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

    def _add_ckpt(self, runner, ckpt_path: str, step: int) -> None:

        if osp.islink(ckpt_path):
            ckpt_path = osp.join(runner.work_dir, os.readlink(ckpt_path))

        if osp.isfile(ckpt_path):
            self.writer.add_snapshot_file(
                tag=self.run_name,
                snapshot_file_path=ckpt_path,
                iteration=step)

    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, add_mode=False)
        if tags:
            self.writer.add_scalars(
                self.get_mode(runner), tags, self.get_step(runner))

    @master_only
    def after_run(self, runner) -> None:

        if self.add_last_ckpt:
            step = self.get_epoch(runner) if self.by_epoch else self.get_iter(
                runner)
            ckpt_path = osp.join(runner.work_dir, 'latest.pth')
            self._add_ckpt(runner, ckpt_path, step)

        # flush the buffer and send a task ending signal to Pavi
        self.writer.close()

    @master_only
    def before_epoch(self, runner) -> None:
        step = self.get_epoch(runner) if self.by_epoch else self.get_iter(
            runner)
        if self.add_graph and \
            step >= self.add_graph_start and \
                ((step - self.add_graph_start) % self.add_graph_interval == 0):
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
        super().after_train_epoch(runner)
        # Do not use runner.epoch since it starts from 0.
        step = self.get_epoch(runner) if self.by_epoch else self.get_iter(
            runner)

        if self.add_ckpt and \
            step >= self.add_ckpt_start and \
                ((step - self.add_ckpt_start) % self.add_ckpt_interval == 0):

            ckpt_path = osp.join(runner.work_dir, f'epoch_{step}.pth')

            self._add_ckpt(runner, ckpt_path, step)

    @master_only
    def after_train_iter(self, runner) -> None:
        super().after_train_iter(runner)

        step = self.get_epoch(runner) if self.by_epoch else self.get_iter(
            runner)

        if self.add_ckpt and \
            step >= self.add_ckpt_start and \
                ((step - self.add_ckpt_start) % self.add_ckpt_interval == 0):

            ckpt_path = osp.join(runner.work_dir, f'iter_{step}.pth')

            self._add_ckpt(runner, ckpt_path, step)
