# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from mmcv.utils import scandir
from ...dist_utils import master_only
from ..hook import HOOKS
from .base import LoggerHook


@HOOKS.register_module()
class WandbLoggerHook(LoggerHook):
    """Class to log metrics with wandb.

    It requires `wandb`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the initialization keys. Check
            https://docs.wandb.ai/ref/python/init for more init arguments.
        interval (int): Logging interval (every k iterations).
            Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
            Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        commit (bool): Save the metrics dict to the wandb server and increment
            the step. If false ``wandb.log`` just updates the current metrics
            dict with the row argument and metrics won't be saved until
            ``wandb.log`` is called with ``commit=True``.
            Default: True.
        by_epoch (bool): Whether EpochBasedRunner is used.
            Default: True.
        with_step (bool): If True, the step will be logged from
            ``self.get_iters``. Otherwise, step will not be logged.
            Default: True.
        log_artifact (bool): If True, artifacts in {work_dir} will be uploaded
            to wandb after training ends.
            Default: True
            `New in version 1.4.3.`
        out_suffix (str or tuple[str], optional): Those filenames ending with
            ``out_suffix`` will be uploaded to wandb.
            Default: ('.log.json', '.log', '.py').
            `New in version 1.4.3.`
        best_log_dict:
            If specified like: {'accuracy': 'max', 'loss': 'min'}, then maximum
             of accuracies and minimum of losses will be logged to wandb.

    .. _wandb:
        https://docs.wandb.ai
    """

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True,
                 log_artifact=True,
                 out_suffix=('.log.json', '.log', '.py'),
                 best_log_dict={}):
        super(WandbLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag, by_epoch)
        self.import_wandb()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step
        self.log_artifact = log_artifact
        self.out_suffix = out_suffix
        self.best_log_dict = best_log_dict
        self.best_metrics = {}

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    @master_only
    def before_run(self, runner):
        super(WandbLoggerHook, self).before_run(runner)
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)

        # log best metric
        for k, v in list(tags.items()):
            for bk, bv in self.best_log_dict.items():
                if bk in k:
                    if (k not in self.best_metrics) or (
                        (bv == 'max') and (v > self.best_metrics[k])) or (
                            (bv == 'min') and (v < self.best_metrics[k])):
                        self.best_metrics[k] = v
                        tags[f'{k}_best'] = v

        if tags:
            if self.with_step:
                self.wandb.log(
                    tags, step=self.get_iter(runner), commit=self.commit)
            else:
                tags['global_step'] = self.get_iter(runner)
                self.wandb.log(tags, commit=self.commit)

    @master_only
    def after_run(self, runner):
        if self.log_artifact:
            wandb_artifact = self.wandb.Artifact(
                name='artifacts', type='model')
            for filename in scandir(runner.work_dir, self.out_suffix, True):
                local_filepath = osp.join(runner.work_dir, filename)
                wandb_artifact.add_file(local_filepath)
            self.wandb.log_artifact(wandb_artifact)
        self.wandb.join()
