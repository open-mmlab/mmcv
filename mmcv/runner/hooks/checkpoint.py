# Copyright (c) Open-MMLab. All rights reserved.
import os

from ..dist_utils import master_only
from .hook import HOOKS, Hook


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 **kwargs):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.args = kwargs

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        runner.logger.info(f'Saving checkpoint at {runner.epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
            current_epoch = runner.epoch + 1
            for epoch in range(current_epoch - self.max_keep_ckpts, 0, -1):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(epoch))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    @master_only
    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return

        runner.logger.info(
            f'Saving checkpoint at {runner.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = runner.work_dir
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            filename_tmpl = self.args.get('filename_tmpl', 'iter_{}.pth')
            current_iter = runner.iter + 1
            for _iter in range(
                    current_iter - self.max_keep_ckpts * self.interval, 0,
                    -self.interval):
                ckpt_path = os.path.join(self.out_dir,
                                         filename_tmpl.format(_iter))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break
