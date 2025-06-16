# Copyright (c) Open-MMLab. All rights reserved.
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from ..hook import Hook
import os
import os.path as osp


class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        by_epoch (bool): Whether EpochBasedRunner is used.
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch
        self.best_ckpt_path = None
        self.best_pck_path = None
        self.best_loss_path = None
        self.key_indicator = 'PCK'  # Esto esta puesto a mano cuidado

    @abstractmethod
    def log(self, runner):
        pass

    @staticmethod
    def is_scalar(val, include_np=True, include_torch=True):
        """Tell the input variable is a scalar or not.

        Args:
            val: Input variable.
            include_np (bool): Whether include 0-d np.ndarray as a scalar.
            include_torch (bool): Whether include 0-d torch.Tensor as a scalar.

        Returns:
            bool: True or False.
        """
        if isinstance(val, numbers.Number):
            return True
        elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
            return True
        elif include_torch and isinstance(val, torch.Tensor) and len(val) == 1:
            return True
        else:
            return False

    def get_mode(self, runner):
        if runner.mode == 'train':
            if 'time' in runner.log_buffer.output:
                mode = 'train'
            else:
                mode = 'val'
        elif runner.mode == 'val':
            mode = 'val'
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return mode

    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch

    def get_iter(self, runner, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = runner.inner_iter + 1
        else:
            current_iter = runner.iter + 1
        return current_iter

    def get_lr_tags(self, runner):
        tags = {}
        lrs = runner.current_lr()
        if isinstance(lrs, dict):
            for name, value in lrs.items():
                tags[f'learning_rate/{name}'] = value[0]
        else:
            tags['learning_rate'] = lrs[0]
        return tags

    def get_momentum_tags(self, runner):
        tags = {}
        momentums = runner.current_momentum()
        if isinstance(momentums, dict):
            for name, value in momentums.items():
                tags[f'momentum/{name}'] = value[0]
        else:
            tags['momentum'] = momentums[0]
        return tags

    def get_loggable_tags(self,
                          runner,
                          allow_scalar=True,
                          allow_text=False,
                          add_mode=True,
                          tags_to_skip=('time', 'data_time')):
        tags = {}
        for var, val in runner.log_buffer.output.items():
            if var in tags_to_skip:
                continue
            if self.is_scalar(val) and not allow_scalar:
                continue
            if isinstance(val, str) and not allow_text:
                continue
            if add_mode:
                var = f'{self.get_mode(runner)}/{var}'
            tags[var] = val
        tags.update(self.get_lr_tags(runner))
        tags.update(self.get_momentum_tags(runner))
        return tags

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, runner):
        runner.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            runner.log_buffer.average(self.interval)

        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

        runner.log_buffer.average()
        self.log(runner)
        output = runner.log_buffer.output
        runner.t_loss.append(output['loss'])
        runner.pt_loss.append(runner.list_mean(runner.t_loss))
        runner.min_train_loss = runner.min_train_loss if runner.min_train_loss < runner.pt_loss[-1] else runner.pt_loss[-1]
        runner.t_pck.append(output['acc_pose'])
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        output = runner.log_buffer.output

        runner.e_loss.append(output['loss'])
        runner.pe_loss.append(runner.list_mean(runner.e_loss))
        runner.min_prom_eval_loss = runner.min_prom_eval_loss if runner.min_prom_eval_loss < runner.pe_loss[-1] else runner.pe_loss[-1]
        dif = runner.pe_loss[-1] - runner.min_prom_eval_loss
        per = dif/runner.min_prom_eval_loss
        if per > 0.1:
            dif_t = runner.pt_loss[-1] - runner.min_train_loss
            per_t = dif_t / runner.min_train_loss
            if per_t < 0.1:
                runner.patience_count += 1
                print(f"Patience {runner.patience_count} / {runner.max_patience}. Min: {runner.min_prom_eval_loss}; Cur: {runner.pt_loss[-1]}; Per: {per}")

        runner.e_pck.append(output['acc_pose'])
        self.save_if_best(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()

    def save_if_best(self, runner):
        # runner.log_buffer.average()
        output = runner.log_buffer.output
        if runner.max_eval_pck < output['acc_pose']:
            runner.max_eval_pck = output['acc_pose']
            if self.best_pck_path and osp.isfile(self.best_pck_path):
                os.remove(self.best_pck_path)
            current = f'epoch_{runner.epoch}'
            best_ckpt_name = f'best_{self.key_indicator}_{current}.pth'
            self.best_pck_path = osp.join(runner.work_dir, best_ckpt_name)
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(
                f'MAX PCK - Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'MAX PCK - Best {self.key_indicator} is {runner.max_eval_pck:0.4f} at {runner.epoch} epoch.')

        if runner.min_eval_loss > output['loss']:
            runner.min_eval_loss = output['loss']
            if self.best_loss_path and osp.isfile(self.best_loss_path):
                os.remove(self.best_loss_path)
            current = f'epoch_{runner.epoch}'
            best_ckpt_name = f'best_LOSS_{current}.pth'
            self.best_loss_path = osp.join(runner.work_dir, best_ckpt_name)
            runner.save_checkpoint(
                runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(
                f'MIN LOSS - Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(
                f'MIN LOSS - Best LOSS is {runner.min_eval_loss:0.4f} at {runner.epoch} epoch.')
