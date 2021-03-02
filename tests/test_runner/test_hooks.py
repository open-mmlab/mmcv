"""Tests the hooks with runners.

CommandLine:
    pytest tests/test_hooks.py
    xdoctest tests/test_hooks.py zero
"""
import logging
import os.path as osp
import re
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from torch.nn.init import constant_
from torch.utils.data import DataLoader

from mmcv.runner import (CheckpointHook, EMAHook, IterTimerHook,
                         MlflowLoggerHook, PaviLoggerHook, WandbLoggerHook,
                         build_runner)
from mmcv.runner.hooks.lr_updater import CosineRestartLrUpdaterHook


def test_checkpoint_hook():
    """xdoctest -m tests/test_runner/test_hooks.py test_checkpoint_hook."""

    # test epoch based runner
    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner('EpochBasedRunner', max_epochs=1)
    runner.meta = dict()
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(checkpointhook)
    runner.run([loader], [('train', 1)])
    assert runner.meta['hook_msgs']['last_ckpt'] == osp.join(
        runner.work_dir, 'epoch_1.pth')
    shutil.rmtree(runner.work_dir)

    # test iter based runner
    runner = _build_demo_runner(
        'IterBasedRunner', max_iters=1, max_epochs=None)
    runner.meta = dict()
    checkpointhook = CheckpointHook(interval=1, by_epoch=False)
    runner.register_hook(checkpointhook)
    runner.run([loader], [('train', 1)])
    assert runner.meta['hook_msgs']['last_ckpt'] == osp.join(
        runner.work_dir, 'iter_1.pth')
    shutil.rmtree(runner.work_dir)


def test_ema_hook():
    """xdoctest -m tests/test_hooks.py test_ema_hook."""

    class DemoModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=1,
                padding=1,
                bias=True)
            self._init_weight()

        def _init_weight(self):
            constant_(self.conv.weight, 0)
            constant_(self.conv.bias, 0)

        def forward(self, x):
            return self.conv(x).sum()

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    loader = DataLoader(torch.ones((1, 1, 1, 1)))
    runner = _build_demo_runner()
    demo_model = DemoModel()
    runner.model = demo_model
    emahook = EMAHook(momentum=0.1, interval=2, warm_up=100, resume_from=None)
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(emahook, priority='HIGHEST')
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_1.pth')
    contain_ema_buffer = False
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            contain_ema_buffer = True
            assert value.sum() == 0
            value.fill_(1)
        else:
            assert value.sum() == 0
    assert contain_ema_buffer
    torch.save(checkpoint, f'{runner.work_dir}/epoch_1.pth')
    work_dir = runner.work_dir
    resume_ema_hook = EMAHook(
        momentum=0.5, warm_up=0, resume_from=f'{work_dir}/epoch_1.pth')
    runner = _build_demo_runner(max_epochs=2)
    runner.model = demo_model
    runner.register_hook(resume_ema_hook, priority='HIGHEST')
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_2.pth')
    contain_ema_buffer = False
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            contain_ema_buffer = True
            assert value.sum() == 2
        else:
            assert value.sum() == 1
    assert contain_ema_buffer
    shutil.rmtree(runner.work_dir)
    shutil.rmtree(work_dir)


def test_pavi_hook():
    sys.modules['pavi'] = MagicMock()

    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.meta = dict(config_dict=dict(lr=0.02, gpu_ids=range(1)))
    hook = PaviLoggerHook(add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    hook.writer.add_scalars.assert_called_with('val', {
        'learning_rate': 0.02,
        'momentum': 0.95
    }, 1)
    hook.writer.add_snapshot_file.assert_called_with(
        tag=runner.work_dir.split('/')[-1],
        snapshot_file_path=osp.join(runner.work_dir, 'epoch_1.pth'),
        iteration=1)


def test_sync_buffers_hook():
    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.register_hook_from_cfg(dict(type='SyncBuffersHook'))
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)


def test_momentum_runner_hook():
    """xdoctest -m tests/test_hooks.py test_momentum_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum scheduler
    hook_cfg = dict(
        type='CyclicMomentumUpdaterHook',
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook_from_cfg(hook_cfg)

    # add momentum LR scheduler
    hook_cfg = dict(
        type='CyclicLrUpdaterHook',
        by_epoch=False,
        target_ratio=(10, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.01999999999999999,
            'momentum': 0.95
        }, 1),
        call('train', {
            'learning_rate': 0.2,
            'momentum': 0.85
        }, 5),
        call('train', {
            'learning_rate': 0.155,
            'momentum': 0.875
        }, 7),
    ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


def test_cosine_runner_hook():
    """xdoctest -m tests/test_hooks.py test_cosine_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum scheduler

    hook_cfg = dict(
        type='CosineAnnealingMomentumUpdaterHook',
        min_momentum_ratio=0.99 / 0.95,
        by_epoch=False,
        warmup_iters=2,
        warmup_ratio=0.9 / 0.95)
    runner.register_hook_from_cfg(hook_cfg)

    # add momentum LR scheduler
    hook_cfg = dict(
        type='CosineAnnealingLrUpdaterHook',
        by_epoch=False,
        min_lr_ratio=0,
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())
    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.02,
            'momentum': 0.95
        }, 1),
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.97
        }, 6),
        call('train', {
            'learning_rate': 0.0004894348370484647,
            'momentum': 0.9890211303259032
        }, 10)
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
        runner.run([loader], [('train', 1)])
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
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    calls = [
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.95
        }, 1),
        call('train', {
            'learning_rate': 0.01,
            'momentum': 0.95
        }, 6),
        call('train', {
            'learning_rate': 0.0009549150281252633,
            'momentum': 0.95
        }, 10)
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
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    hook.mlflow.set_experiment.assert_called_with('test')
    hook.mlflow.log_metrics.assert_called_with(
        {
            'learning_rate': 0.02,
            'momentum': 0.95
        }, step=6)
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
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    hook.wandb.init.assert_called_with()
    hook.wandb.log.assert_called_with({
        'learning_rate': 0.02,
        'momentum': 0.95
    },
                                      step=6,
                                      commit=True)
    hook.wandb.join.assert_called_with()


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None):

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
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


def test_runner_with_revise_keys():

    import os

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 3, 1)

    class PrefixModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.backbone = Model()

    pmodel = PrefixModel()
    model = Model()
    checkpoint_path = os.path.join(tempfile.gettempdir(), 'checkpoint.pth')

    # add prefix
    torch.save(model.state_dict(), checkpoint_path)
    runner = _build_demo_runner(runner_type='EpochBasedRunner')
    runner.model = pmodel
    state_dict = runner.load_checkpoint(
        checkpoint_path, revise_keys=[(r'^', 'backbone.')])
    for key in pmodel.backbone.state_dict().keys():
        assert torch.equal(pmodel.backbone.state_dict()[key], state_dict[key])
    # strip prefix
    torch.save(pmodel.state_dict(), checkpoint_path)
    runner.model = model
    state_dict = runner.load_checkpoint(
        checkpoint_path, revise_keys=[(r'^backbone\.', '')])
    for key in state_dict.keys():
        key_stripped = re.sub(r'^backbone\.', '', key)
        assert torch.equal(model.state_dict()[key_stripped], state_dict[key])
    os.remove(checkpoint_path)
