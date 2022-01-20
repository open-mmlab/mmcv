# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from functools import partial

import torch
from torch.utils.data import IterableDataset

from ....parallel import DataContainer
from ....parallel.utils import is_module_wrapper
from ....utils import TORCH_VERSION, digit_version
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class TensorboardLoggerHook(LoggerHook):
    """Class to visualize model and log metrics to Tensorboard.

    Args:
        log_dir (string): Save directory location. Default: None. If default
            values are used, directory location is ``runner.work_dir``/tf_logs.
        interval (int): Logging interval (every k iterations). Default: True.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
        add_graph (bool): Whether to visualize model. Default: False.
            `New in version 1.4.4.`
        img_key (string): Used to visualize model, get image data from Dataset.
            Default: 'img'.
            `New in version 1.4.4.`
        img_metas (List[List[dict]]): Used to visualize model, the outer list
            indicates test-time augs (multiscale, flip, etc.) and the inner
            list indicates images in a batch.
            `New in version 1.4.4.`
        verbose (bool): Used to visualize model, Whether to print graph
            structure in console. Default: False.
            `New in version 1.4.4.`
    """

    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 add_graph=False,
                 img_key='img',
                 img_metas=None,
                 verbose=False):
        super(TensorboardLoggerHook, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        self.add_graph = add_graph
        self.img_key = img_key
        self.img_metas = img_metas
        self.verbose = verbose

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        self.writer.close()

    def visual_model(self, runner):
        from ....runner import IterLoader
        if is_module_wrapper(runner.model):
            _model = runner.model.module
        else:
            _model = runner.model
        device = next(_model.parameters()).device
        if isinstance(runner.data_loader, IterLoader):
            dataloader = runner.data_loader._dataloader
        else:
            dataloader = runner.data_loader
        if isinstance(dataloader.dataset, IterableDataset):
            data = next(iter(dataloader))
        else:
            data = dataloader.dataset[0]
        image = data[self.img_key]
        if isinstance(image, DataContainer):
            image = image.data
        image = image.to(device).unsqueeze(0)
        origin_forward = _model.forward
        _model.forward = partial(
            _model.forward, img_metas=self.img_metas, return_loss=False)
        with torch.no_grad():
            self.writer.add_graph(_model, image, verbose=self.verbose)
        _model.forward = origin_forward

    @master_only
    def before_epoch(self, runner):
        if runner.epoch == 0 and self.add_graph:
            # avoid circular dependencies
            from ....runner import EpochBasedRunner
            if isinstance(runner, EpochBasedRunner):
                self.visual_model(runner)

    @master_only
    def before_iter(self, runner):
        if runner.iter == 0 and self.add_graph:
            # avoid circular dependencies
            from ....runner import IterBasedRunner
            if isinstance(runner, IterBasedRunner):
                self.visual_model(runner)
