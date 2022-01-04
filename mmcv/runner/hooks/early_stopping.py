# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/early_stopping.py

import warnings
from typing import Callable, Dict, Optional

import numpy as np

from ..hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    r"""Stop training when a monitored metric has stopped improving.

    Args:
        monitor: quantity to be monitored.
        phase: runner mode when executing model.Evaluation will only take place
            with corresponding runner mode.
        min_delta: minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than or equal to
            `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which training will be stopped. Under the default
            configuration ,one check happens after every training epoch.
            However, the frequency of validation can be modified by setting
            various parameters on the
            ``Runner``, for example ``[(train, 10), (val, 1)]``.
            .. note::
                It must be noted that the patience parameter counts the number
                of validation checks with no improvement, and not the number of
                training epochs. Therefore, with parameters
                ``[(train, 10), (val, 1)]`` and ``patience=3``, the runner
                will perform at least 40 training epochs before being stopped.
        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will
            stop when the quantity monitored has stopped decreasing and in
            ``'max'`` mode it will stop when the quantity monitored has
            stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the
            validation metrics.
        check_finite: When set ``True``, stops training when the monitor
            becomes NaN or infinite.
        stopping_threshold: Stop training immediately once the monitored
            quantity reaches this threshold.
        divergence_threshold: Stop training as soon as the monitored quantity
            becomes worse than this threshold.

    Example:
        >>> EarlyStoppingHook('loss', mode='min', divergence_threshold=100)
        >>> # if loss > 100, training will be stopped
        >>> EarlyStoppingHook('loss', mode='min', stopping_treshold=0.01)
        >>> # if loss < 0.01, training will be stopped regardless of patience.
    """

    mode_dict = {'min': np.less, 'max': np.greater}

    def __init__(self,
                 monitor: str,
                 phase: str = 'train',
                 min_delta: float = 0.0,
                 patience: int = 3,
                 verbose: bool = False,
                 mode: str = 'min',
                 strict: bool = True,
                 check_finite: bool = True,
                 stopping_threshold: Optional[float] = None,
                 divergence_threshold: Optional[float] = None,
                 by_epoch=True):
        assert isinstance(
            patience,
            int) and patience > 0, 'patience must be positive integer'
        self.monitor = monitor
        self.phase = phase
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.by_epoch = by_epoch
        self.wait_count = 0
        self.stopped_step = 0

        if self.mode not in self.mode_dict:
            raise ValueError(
                f"`mode` can be {', '.join(self.mode_dict.keys())}, got"
                f' {self.mode}')

        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    def before_run(self, runner):
        if runner.meta is None:
            warnings.warn('runner.meta is None. Creating an empty one.')
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())
        self.wait_count = runner.meta['hook_msgs'].get('wait_count',
                                                       self.wait_count)
        self.best_score = runner.meta['hook_msgs'].get('early_stop_best_score',
                                                       self.best_score)

    def before_train_epoch(self, runner):
        runner.log_buffer.clear()

    def before_val_epoch(self, runner):
        runner.log_buffer.clear()

    def after_train_iter(self, runner):
        if self.by_epoch or self.phase == 'val':
            return
        runner.log_buffer.average(1)
        self._run_early_stopping_check(runner, runner.log_buffer.output)

    def after_train_epoch(self, runner):
        if not self.by_epoch or self.phase == 'val':
            return
        runner.log_buffer.average()
        self._run_early_stopping_check(runner, runner.log_buffer.output)

    def after_val_iter(self, runner):
        if self.by_epoch or self.phase == 'train':
            return
        runner.log_buffer.average(1)
        self._run_early_stopping_check(runner, runner.log_buffer.output)

    def after_val_epoch(self, runner):
        if not self.by_epoch or self.phase == 'train':
            return
        runner.log_buffer.average()
        self._run_early_stopping_check(runner, runner.log_buffer.output)

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def get_mode(self, runner):
        return runner.mode

    def _validate_condition_metric(self, runner, logs: Dict[str,
                                                            float]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f'Early stopping conditioned on metric `{self.monitor}` which is'
            ' not available.  Pass in or modify your `EarlyStopping` callback'
            ' to use any of the following:'
            f' `{"`, `".join(list(logs.keys()))}`')

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                runner.logger.warn(error_msg)

            return False

        return True

    def _run_early_stopping_check(self, runner, logs: Dict[str, float]):
        if not self._validate_condition_metric(
                runner, logs):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)
        runner.should_stop = runner.should_stop or should_stop
        runner.meta['hook_msgs']['wait_count'] = self.wait_count
        runner.meta['hook_msgs']['early_stop_best_score'] = self.best_score
        if reason and self.verbose > 0:
            runner.logger.info(reason)

    def _evaluate_stopping_criteria(self, current: float):
        should_stop = False
        reason = None

        if self.check_finite and not np.isfinite(current):
            should_stop = True
            reason = (
                f'Monitored metric {self.monitor} = {current} is not finite.'
                f' Previous best value was {self.best_score:.3f}. Signaling'
                ' runner to stop.')
        elif self.stopping_threshold is not None and self.monitor_op(
                current, self.stopping_threshold):
            should_stop = True
            reason = (
                'Stopping threshold reached:'
                f' {self.monitor} = {current} {self.mode_dict[self.mode]}'
                f' {self.stopping_threshold}.'
                ' Signaling Trainer to stop.')
        elif self.divergence_threshold is not None and self.monitor_op(
                -current, -self.divergence_threshold):
            should_stop = True
            reason = (
                'Divergence threshold reached:'
                f' {self.monitor} = {current} {self.mode_dict[self.mode]}'
                f' {self.divergence_threshold}.'
                ' Signaling Trainer to stop.')
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f'Monitored metric {self.monitor} did not improve in the'
                    f' last {self.wait_count} records.'
                    f' Best score: {self.best_score:.3f}. Signaling Runner to'
                    ' stop.')

        return should_stop, reason

    def _improvement_message(self, current: np.float) -> str:
        """Formats a log message that informs the user about an improvement in
        the monitored score."""
        if np.isfinite(self.best_score):
            msg = (f'Metric {self.monitor} improved by'
                   f' {abs(self.best_score - current):.3f} >='
                   f' min_delta = {abs(self.min_delta)}. New best score:'
                   f' {current:.3f}')
        else:
            msg = (f'Metric {self.monitor} improved. New best score:'
                   f' {current:.3f}')
        return msg
