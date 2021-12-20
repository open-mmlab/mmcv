"""Tests the hooks with runners.

CommandLine:
    pytest tests/test_runner/test_early_stop.py
"""
import tempfile
from typing import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.runner import EarlyStoppingHook, EpochBasedRunner, IterBasedRunner
from mmcv.utils import get_logger


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.step_train = 0
        self.step_val = 0

    def forward(self, x, **kwargs):
        return x

    def train_step(self, data_batch, optimizer, **kwargs):
        log = {
            'log_vars': {
                self.loss_key:
                torch.tensor(self.loss_values[self.step], dtype=torch.float32)
            },
            'num_samples': data_batch.size(0)
        }
        self.step += 1
        return log

    def val_step(self, data_batch, optimizer, **kwargs):
        return {'loss': torch.sum(self(data_batch['x']))}


def _build_epoch_runner(model, steps=4):
    tmp_dir = tempfile.mkdtemp()

    runner = EpochBasedRunner(
        model=model,
        work_dir=tmp_dir,
        logger=get_logger('demo'),
        max_epochs=steps)
    return runner


def _build_iter_runner(model, steps=4):
    tmp_dir = tempfile.mkdtemp()

    runner = IterBasedRunner(
        model=model,
        work_dir=tmp_dir,
        logger=get_logger('demo'),
        max_iters=steps)
    return runner


def test_early_stop():
    with pytest.raises(AssertionError):
        # `patience` should be a positive integer
        EarlyStoppingHook('loss', patience=-1)


@pytest.mark.parametrize('_build_demo_runner, by_epoch, step_attr',
                         [(_build_epoch_runner, True, 'epoch'),
                          (_build_iter_runner, False, 'iter')])
@pytest.mark.parametrize('loss_values, patience, expected_stop_step',
                         [([6, 5, 5, 5, 5, 5], 3, 5),
                          ([6, 5, 4, 4, 3, 3], 1, 4),
                          ([6, 5, 6, 5, 5, 5], 3, 5)])
@pytest.mark.parametrize('workflow, phase', [([('train', 1)], 'train'),
                                             ([('train', 1),
                                               ('val', 1)], 'val')])
def test_early_stopping_patience(_build_demo_runner: Callable, by_epoch: int,
                                 step_attr: str, loss_values: list,
                                 patience: int, expected_stop_step: int,
                                 workflow: list, phase: str):
    if by_epoch:
        # simulate 2 ite / epoch.
        loss_values = np.array(loss_values).repeat(2)
    else:
        loss_values = np.array(loss_values)

    class Model(BaseModel):

        def train_step(self, data_batch, optimizer, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_train], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_train += 1
            return log

        def val_step(self, data_batch, *args, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_val], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_val += 1
            return log

    dataloader = DataLoader(torch.ones(10, 2), batch_size=5)
    runner = _build_demo_runner(Model(), steps=10)
    hook = EarlyStoppingHook(
        monitor='loss', phase=phase, patience=patience, by_epoch=by_epoch)
    runner.register_hook(hook)
    runner.run([dataloader] * len(workflow), workflow)
    assert getattr(runner, step_attr) == expected_stop_step


@pytest.mark.parametrize('_build_demo_runner, by_epoch',
                         [(_build_epoch_runner, True),
                          (_build_iter_runner, False)])
def test_early_stopping_no_monitor(_build_demo_runner, by_epoch: bool):
    """Test that early stopping callback falls back to training metrics when no
    validation defined."""

    class Model(BaseModel):

        def train_step(self, data_batch, optimizer, **kwargs):
            log = {
                'log_vars': {
                    'loss': torch.tensor(0, dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_train += 1
            return log

    dataloader = DataLoader(torch.ones(10, 2), batch_size=5)
    runner = _build_demo_runner(Model(), steps=10)
    hook = EarlyStoppingHook(
        monitor='UNKNOWN_KEY', patience=3, by_epoch=by_epoch)
    runner.register_hook(hook)
    with pytest.raises(RuntimeError):
        runner.run([dataloader], [('train', 1)])


@pytest.mark.parametrize('_build_demo_runner, by_epoch, step_attr',
                         [(_build_epoch_runner, True, 'epoch'),
                          (_build_iter_runner, False, 'iter')])
@pytest.mark.parametrize(
    'stopping_threshold,divergence_threshold,loss_values,expected_stop_step',
    [
        (None, None, [8, 4, 2, 3, 4, 5, 8, 10], 6),
        (2.9, None, [9, 8, 7, 6, 5, 6, 4, 3, 2, 1], 9),
        (None, 15.9, [9, 4, 2, 16, 32, 64], 4),
    ],
)
@pytest.mark.parametrize('workflow, phase', [([('train', 1)], 'train'),
                                             ([('train', 1),
                                               ('val', 1)], 'val')])
def test_early_stopping_thresholds(_build_demo_runner, by_epoch: bool,
                                   step_attr: str, stopping_threshold: int,
                                   divergence_threshold: int,
                                   loss_values: list, expected_stop_step: int,
                                   workflow: tuple, phase: str):
    if by_epoch:
        # simulate 2 ite / epoch.
        loss_values = np.array(loss_values).repeat(2)
    else:
        loss_values = np.array(loss_values)

    class Model(BaseModel):

        def train_step(self, data_batch, optimizer, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_train], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_train += 1
            return log

        def val_step(self, data_batch, *args, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_val], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_val += 1
            return log

    dataloader = DataLoader(torch.ones(10, 2), batch_size=5)
    runner = _build_demo_runner(Model(), steps=10)
    hook = EarlyStoppingHook(
        monitor='loss',
        patience=3,
        by_epoch=by_epoch,
        stopping_threshold=stopping_threshold,
        divergence_threshold=divergence_threshold)
    runner.register_hook(hook)
    runner.run([dataloader] * len(workflow), workflow)
    assert getattr(runner, step_attr) == expected_stop_step


@pytest.mark.parametrize('_build_demo_runner, by_epoch, step_attr',
                         [(_build_epoch_runner, True, 'epoch'),
                          (_build_iter_runner, False, 'iter')])
@pytest.mark.parametrize(
    'stop_value',
    [torch.tensor(np.inf), torch.tensor(np.nan)])
def test_early_stopping_on_non_finite_monitor(_build_demo_runner,
                                              by_epoch: bool, step_attr: str,
                                              stop_value):

    loss_values = [4, 3, stop_value, 2, 1]
    expected_stop_step = 3

    if by_epoch:
        # simulate 2 ite / epoch.
        loss_values = np.array(loss_values).repeat(2)
    else:
        loss_values = np.array(loss_values)

    class Model(BaseModel):

        def train_step(self, data_batch, optimizer, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_train], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_train += 1
            return log

        def val_step(self, data_batch, *args, **kwargs):
            log = {
                'log_vars': {
                    'loss':
                    torch.tensor(
                        loss_values[self.step_val], dtype=torch.float32)
                },
                'num_samples': data_batch.size(0)
            }
            self.step_val += 1
            return log

    dataloader = DataLoader(torch.ones(10, 2), batch_size=5)
    runner = _build_demo_runner(Model(), steps=10)
    hook = EarlyStoppingHook(monitor='loss', patience=3, by_epoch=by_epoch)
    runner.register_hook(hook)
    runner.run([dataloader], [('train', 1)])
    assert getattr(runner, step_attr) == expected_stop_step
