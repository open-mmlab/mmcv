"""
Tests the hooks with runners.

CommandLine:
    pytest tests/test_hooks.py
    xdoctest tests/test_hooks.py zero

"""
import logging
import os.path as osp
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.runner import (EpochBasedRunner, IterTimerHook, MlflowLoggerHook,
                         PaviLoggerHook, WandbLoggerHook)
from mmcv.runner.hooks.lr_updater import (CosineAnealingLrUpdaterHook,
                                          CosineRestartLrUpdaterHook,
                                          CyclicLrUpdaterHook)
from mmcv.runner.hooks.momentum_updater import (
    CosineAnealingMomentumUpdaterHook, CyclicMomentumUpdaterHook)


def test_pavi_hook():
    sys.modules['pavi'] = MagicMock()

    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    hook = PaviLoggerHook(add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    hook.writer.add_scalars.assert_called_with('val', {
        'learning_rate': 0.02,
        'momentum': 0.95
    }, 5)
    hook.writer.add_snapshot_file.assert_called_with(
        tag=runner.work_dir.split('/')[-1],
        snapshot_file_path=osp.join(runner.work_dir, 'latest.pth'),
        iteration=5)


def test_momentum_runner_hook():
    """
    xdoctest -m tests/test_hooks.py test_momentum_runner_hook
    """
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum scheduler
    hook = CyclicMomentumUpdaterHook(
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)

    # add momentum LR scheduler
    hook = CyclicLrUpdaterHook(
        by_epoch=False,
        target_ratio=(10, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)], 1)
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.01999999999999999,
            'momentum': 0.95
        }, 0),
        call('train', {
            'learning_rate': 0.2,
            'momentum': 0.85
        }, 4),
        call('train', {
            'learning_rate': 0.155,
            'momentum': 0.875
        }, 6),
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


def test_cosine_runner_hook():
    """
    xdoctest -m tests/test_hooks.py test_cosine_runner_hook
    """
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum scheduler
    hook = CosineAnealingMomentumUpdaterHook(
        min_momentum_ratio=0.99 / 0.95,
        by_epoch=False,
        warmup_iters=2,
        warmup_ratio=0.9 / 0.95)
    runner.register_hook(hook)

    # add momentum LR scheduler
    hook = CosineAnealingLrUpdaterHook(
        by_epoch=False, min_lr_ratio=0, warmup_iters=2, warmup_ratio=0.9)
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)], 1)
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.02,
            'momentum': 0.95
        }, 0),
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.97
        }, 5),
        call('train', {
            'learning_rate': 0.0004894348370484647,
            'momentum': 0.9890211303259032
        }, 9)
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


def test_cosine_restart_lr_update_hook():
    """Test CosineRestartLrUpdaterHook."""
    with pytest.raises(AssertionError):
        # either `min_lr` or `min_lr_ratio` should be specified
        CosineRestartLrUpdaterHook(
            by_epoch=False,
            periods=[2, 10],
            restart_weights=[0.5, 0.5],
            min_lr=0.1,
            min_lr_ratio=0)

    with pytest.raises(AssertionError):
        # periods and restart_weights should have the same length
        CosineRestartLrUpdaterHook(
            by_epoch=False,
            periods=[2, 10],
            restart_weights=[0.5],
            min_lr_ratio=0)

    with pytest.raises(ValueError):
        # the last cumulative_periods 7 (out of [5, 7]) should >= 10
        sys.modules['pavi'] = MagicMock()
        loader = DataLoader(torch.ones((10, 2)))
        runner = _build_demo_runner()

        # add cosine restart LR scheduler
        hook = CosineRestartLrUpdaterHook(
            by_epoch=False,
            periods=[5, 2],  # cumulative_periods [5, 7 (5 + 2)]
            restart_weights=[0.5, 0.5],
            min_lr=0.0001)
        runner.register_hook(hook)
        runner.register_hook(IterTimerHook())

        # add pavi hook
        hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
        runner.register_hook(hook)
        runner.run([loader], [('train', 1)], 1)
        shutil.rmtree(runner.work_dir)

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add cosine restart LR scheduler
    hook = CosineRestartLrUpdaterHook(
        by_epoch=False,
        periods=[5, 5],
        restart_weights=[0.5, 0.5],
        min_lr_ratio=0)
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)], 1)
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.95
        }, 0),
        call('train', {
            'learning_rate': 0.0,
            'momentum': 0.95
        }, 5),
        call('train', {
            'learning_rate': 0.0009549150281252633,
            'momentum': 0.95
        }, 9)
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize('log_model', (True, False))
def test_mlflow_hook(log_model):
    sys.modules['mlflow'] = MagicMock()
    sys.modules['mlflow.pytorch'] = MagicMock()

    runner = _build_demo_runner()
    loader = DataLoader(torch.ones((5, 2)))

    hook = MlflowLoggerHook(exp_name='test', log_model=log_model)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)
    shutil.rmtree(runner.work_dir)

    hook.mlflow.set_experiment.assert_called_with('test')
    hook.mlflow.log_metrics.assert_called_with(
        {
            'learning_rate': 0.02,
            'momentum': 0.95
        }, step=5)
    if log_model:
        hook.mlflow_pytorch.log_model.assert_called_with(
            runner.model, 'models')
    else:
        assert not hook.mlflow_pytorch.log_model.called


def test_wandb_hook():
    sys.modules['wandb'] = MagicMock()
    runner = _build_demo_runner()
    hook = WandbLoggerHook()
    loader = DataLoader(torch.ones((5, 2)))

    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)
    shutil.rmtree(runner.work_dir)

    hook.wandb.init.assert_called_with()
    hook.wandb.log.assert_called_with({
        'learning_rate': 0.02,
        'momentum': 0.95
    },
                                      step=5)
    hook.wandb.join.assert_called_with()


def _build_demo_runner():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    tmp_dir = tempfile.mkdtemp()
    runner = EpochBasedRunner(
        model=model,
        work_dir=tmp_dir,
        optimizer=optimizer,
        logger=logging.getLogger())

    runner.register_logger_hooks(log_config)
    return runner
