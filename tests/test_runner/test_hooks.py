"""Tests the hooks with runners.

CommandLine:
    pytest tests/test_runner/test_hooks.py
    xdoctest tests/test_hooks.py zero
"""
import logging
import os.path as osp
import platform
import random
import re
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import torch
import torch.nn as nn
from tests.pytest_util import mock_package
from torch.nn.init import constant_
from torch.utils.data import DataLoader

from mmcv.fileio.file_client import FileClient, PetrelBackend
from mmcv.runner import (CheckpointHook, DvcliveLoggerHook, EMAHook,
                         Fp16OptimizerHook,
                         GradientCumulativeFp16OptimizerHook,
                         GradientCumulativeOptimizerHook, IterTimerHook,
                         MlflowLoggerHook, NeptuneLoggerHook, OptimizerHook,
                         PaviLoggerHook, WandbLoggerHook, build_runner)
from mmcv.runner.fp16_utils import auto_fp16
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks.lr_updater import (CosineRestartLrUpdaterHook,
                                          CyclicLrUpdaterHook,
                                          FlatCosineAnnealingLrUpdaterHook,
                                          OneCycleLrUpdaterHook,
                                          StepLrUpdaterHook)


def test_optimizerhook():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1)
            self.conv2 = nn.Conv2d(
                in_channels=2,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1)
            self.conv3 = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x1, x2

    model = Model()
    x = torch.rand(1, 1, 3, 3)

    dummy_runner = Mock()
    dummy_runner.optimizer.zero_grad = Mock(return_value=None)
    dummy_runner.optimizer.step = Mock(return_value=None)
    dummy_runner.model = model
    dummy_runner.outputs = dict()

    dummy_runner.outputs['num_samples'] = 0

    class DummyLogger():

        def __init__(self):
            self.msg = ''

        def log(self, msg=None, **kwargs):
            self.msg += msg

    dummy_runner.logger = DummyLogger()
    optimizer_hook = OptimizerHook(
        dict(max_norm=2), detect_anomalous_params=True)

    dummy_runner.outputs['loss'] = model(x)[0].sum()
    optimizer_hook.after_train_iter(dummy_runner)
    # assert the parameters of conv2 and conv3 are not in the
    # computational graph which is with x1.sum() as root.
    assert 'conv2.weight' in dummy_runner.logger.msg
    assert 'conv2.bias' in dummy_runner.logger.msg
    assert 'conv3.weight' in dummy_runner.logger.msg
    assert 'conv3.bias' in dummy_runner.logger.msg
    assert 'conv1.weight' not in dummy_runner.logger.msg
    assert 'conv1.bias' not in dummy_runner.logger.msg

    dummy_runner.outputs['loss'] = model(x)[1].sum()
    dummy_runner.logger.msg = ''
    optimizer_hook.after_train_iter(dummy_runner)
    # assert the parameters of conv3 are not in the computational graph
    assert 'conv3.weight' in dummy_runner.logger.msg
    assert 'conv3.bias' in dummy_runner.logger.msg
    assert 'conv2.weight' not in dummy_runner.logger.msg
    assert 'conv2.bias' not in dummy_runner.logger.msg
    assert 'conv1.weight' not in dummy_runner.logger.msg
    assert 'conv1.bias' not in dummy_runner.logger.msg


def test_checkpoint_hook(tmp_path):
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

    # test petrel oss when the type of runner is `EpochBasedRunner`
    runner = _build_demo_runner('EpochBasedRunner', max_epochs=4)
    runner.meta = dict()
    out_dir = 's3://user/data'
    with mock_package('petrel_client', 'petrel_client.client'), \
         patch.object(PetrelBackend, 'put') as mock_put, \
         patch.object(PetrelBackend, 'remove') as mock_remove, \
         patch.object(PetrelBackend, 'isfile') as mock_isfile:
        FileClient._instances = {}
        checkpointhook = CheckpointHook(
            interval=1, out_dir=out_dir, by_epoch=True, max_keep_ckpts=2)
        runner.register_hook(checkpointhook)
        runner.run([loader], [('train', 1)])
        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        assert runner.meta['hook_msgs']['last_ckpt'] == \
               '/'.join([out_dir, basename, 'epoch_4.pth'])
    mock_put.assert_called()
    mock_remove.assert_called()
    mock_isfile.assert_called()
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

    # test petrel oss when the type of runner is `IterBasedRunner`
    runner = _build_demo_runner(
        'IterBasedRunner', max_iters=4, max_epochs=None)
    runner.meta = dict()
    out_dir = 's3://user/data'
    with mock_package('petrel_client', 'petrel_client.client'), \
         patch.object(PetrelBackend, 'put') as mock_put, \
         patch.object(PetrelBackend, 'remove') as mock_remove, \
         patch.object(PetrelBackend, 'isfile') as mock_isfile:
        FileClient._instances = {}
        checkpointhook = CheckpointHook(
            interval=1, out_dir=out_dir, by_epoch=False, max_keep_ckpts=2)
        runner.register_hook(checkpointhook)
        runner.run([loader], [('train', 1)])
        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        assert runner.meta['hook_msgs']['last_ckpt'] == \
               '/'.join([out_dir, basename, 'iter_4.pth'])
    mock_put.assert_called()
    mock_remove.assert_called()
    mock_isfile.assert_called()
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


def test_custom_hook():

    @HOOKS.register_module()
    class ToyHook(Hook):

        def __init__(self, info, *args, **kwargs):
            super().__init__()
            self.info = info

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test if custom_hooks is None
    runner.register_custom_hooks(None)
    assert len(runner.hooks) == 0
    # test if custom_hooks is dict list
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=51, info=51),
        dict(type='ToyHook', priority=49, info=49)
    ]
    runner.register_custom_hooks(custom_hooks_cfg)
    assert [hook.info for hook in runner.hooks] == [49, 51]
    # test if custom_hooks is object and without priority
    runner.register_custom_hooks(ToyHook(info='default'))
    assert len(runner.hooks) == 3 and runner.hooks[1].info == 'default'
    shutil.rmtree(runner.work_dir)

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test custom_hooks with string priority setting
    priority_ranks = [
        'HIGHEST', 'VERY_HIGH', 'HIGH', 'ABOVE_NORMAL', 'NORMAL',
        'BELOW_NORMAL', 'LOW', 'VERY_LOW', 'LOWEST'
    ]
    random_priority_ranks = priority_ranks.copy()
    random.shuffle(random_priority_ranks)
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=rank, info=rank)
        for rank in random_priority_ranks
    ]
    runner.register_custom_hooks(custom_hooks_cfg)
    assert [hook.info for hook in runner.hooks] == priority_ranks
    shutil.rmtree(runner.work_dir)

    runner = _build_demo_runner_without_hook('EpochBasedRunner', max_epochs=1)
    # test register_training_hooks order
    custom_hooks_cfg = [
        dict(type='ToyHook', priority=1, info='custom 1'),
        dict(type='ToyHook', priority='NORMAL', info='custom normal'),
        dict(type='ToyHook', priority=89, info='custom 89')
    ]
    runner.register_training_hooks(
        lr_config=ToyHook('lr'),
        optimizer_config=ToyHook('optimizer'),
        checkpoint_config=ToyHook('checkpoint'),
        log_config=dict(interval=1, hooks=[dict(type='ToyHook', info='log')]),
        momentum_config=ToyHook('momentum'),
        timer_config=ToyHook('timer'),
        custom_hooks_config=custom_hooks_cfg)
    # If custom hooks have same priority with default hooks, custom hooks
    # will be triggered after default hooks.
    hooks_order = [
        'custom 1', 'lr', 'momentum', 'optimizer', 'checkpoint',
        'custom normal', 'timer', 'custom 89', 'log'
    ]
    assert [hook.info for hook in runner.hooks] == hooks_order
    shutil.rmtree(runner.work_dir)


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
    # in Windows environment, the latest checkpoint is copied from epoch_1.pth
    if platform.system() == 'Windows':
        snapshot_file_path = osp.join(runner.work_dir, 'latest.pth')
    else:
        snapshot_file_path = osp.join(runner.work_dir, 'epoch_1.pth')
    hook.writer.add_snapshot_file.assert_called_with(
        tag=runner.work_dir.split('/')[-1],
        snapshot_file_path=snapshot_file_path,
        iteration=1)


def test_sync_buffers_hook():
    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.register_hook_from_cfg(dict(type='SyncBuffersHook'))
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_momentum_runner_hook(multi_optimziers):
    """xdoctest -m tests/test_hooks.py test_momentum_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

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
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.01999999999999999,
                    'learning_rate/model2': 0.009999999999999995,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.2,
                    'learning_rate/model2': 0.1,
                    'momentum/model1': 0.85,
                    'momentum/model2': 0.8052631578947369,
                }, 5),
            call(
                'train', {
                    'learning_rate/model1': 0.155,
                    'learning_rate/model2': 0.0775,
                    'momentum/model1': 0.875,
                    'momentum/model2': 0.8289473684210527,
                }, 7)
        ]
    else:
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

    # test constant momentum warmup
    sys.modules['pavi'] = MagicMock()
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='StepMomentumUpdaterHook',
        by_epoch=False,
        warmup='constant',
        warmup_iters=5,
        warmup_ratio=0.5,
        step=[10],
    )
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))

    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.9,
                    'momentum/model2': 1.8,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.9,
                    'momentum/model2': 1.8,
                }, 5),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 10),
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.9
            }, 1),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.9
            }, 5),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 10),
        ]

    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)

    # test linear momentum warmup
    sys.modules['pavi'] = MagicMock()
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='StepMomentumUpdaterHook',
        by_epoch=False,
        warmup='linear',
        warmup_iters=5,
        warmup_ratio=0.5,
        step=[10],
    )
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))

    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.9,
                    'momentum/model2': 1.8,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.3571428571428572,
                    'momentum/model2': 1.2857142857142858,
                }, 3),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 10),
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.9
            }, 1),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.3571428571428572
            }, 3),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 10),
        ]

    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)

    # test exponentially momentum warmup
    sys.modules['pavi'] = MagicMock()
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='StepMomentumUpdaterHook',
        by_epoch=False,
        warmup='exp',
        warmup_iters=5,
        warmup_ratio=0.5,
        step=[10],
    )
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))

    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.9,
                    'momentum/model2': 1.8,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 1.4399307381848783,
                    'momentum/model2': 1.3641449098593583,
                }, 3),
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 10),
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.9
            }, 1),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 1.4399307381848783
            }, 3),
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 10),
        ]

    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_cosine_runner_hook(multi_optimziers):
    """xdoctest -m tests/test_hooks.py test_cosine_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

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
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.01,
                    'learning_rate/model2': 0.005,
                    'momentum/model1': 0.97,
                    'momentum/model2': 0.9189473684210527,
                }, 6),
            call(
                'train', {
                    'learning_rate/model1': 0.0004894348370484647,
                    'learning_rate/model2': 0.00024471741852423234,
                    'momentum/model1': 0.9890211303259032,
                    'momentum/model2': 0.9369673866245399,
                }, 10)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.01,
                'momentum': 0.97
            }, 6),
            call(
                'train', {
                    'learning_rate': 0.0004894348370484647,
                    'momentum': 0.9890211303259032
                }, 10)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize('multi_optimziers, by_epoch', [(False, False),
                                                        (True, False),
                                                        (False, True),
                                                        (True, True)])
def test_flat_cosine_runner_hook(multi_optimziers, by_epoch):
    """xdoctest -m tests/test_hooks.py test_flat_cosine_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    max_epochs = 10 if by_epoch else 1
    runner = _build_demo_runner(
        multi_optimziers=multi_optimziers, max_epochs=max_epochs)

    with pytest.raises(ValueError):
        # start_percent: expected float between 0 and 1
        FlatCosineAnnealingLrUpdaterHook(start_percent=-0.1, min_lr_ratio=0)

    # add LR scheduler
    hook_cfg = dict(
        type='FlatCosineAnnealingLrUpdaterHook',
        by_epoch=by_epoch,
        min_lr_ratio=0,
        warmup='linear',
        warmup_iters=10 if by_epoch else 2,
        warmup_ratio=0.9,
        start_percent=0.5)
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
    if multi_optimziers:
        if by_epoch:
            calls = [
                call(
                    'train', {
                        'learning_rate/model1': 0.018000000000000002,
                        'learning_rate/model2': 0.009000000000000001,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9,
                    }, 1),
                call(
                    'train', {
                        'learning_rate/model1': 0.02,
                        'learning_rate/model2': 0.01,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9,
                    }, 11),
                call(
                    'train', {
                        'learning_rate/model1': 0.018090169943749474,
                        'learning_rate/model2': 0.009045084971874737,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9,
                    }, 61),
                call(
                    'train', {
                        'learning_rate/model1': 0.0019098300562505265,
                        'learning_rate/model2': 0.0009549150281252633,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9,
                    }, 100)
            ]
        else:
            calls = [
                call(
                    'train', {
                        'learning_rate/model1': 0.018000000000000002,
                        'learning_rate/model2': 0.009000000000000001,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9
                    }, 1),
                call(
                    'train', {
                        'learning_rate/model1': 0.02,
                        'learning_rate/model2': 0.01,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9
                    }, 6),
                call(
                    'train', {
                        'learning_rate/model1': 0.018090169943749474,
                        'learning_rate/model2': 0.009045084971874737,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9
                    }, 7),
                call(
                    'train', {
                        'learning_rate/model1': 0.0019098300562505265,
                        'learning_rate/model2': 0.0009549150281252633,
                        'momentum/model1': 0.95,
                        'momentum/model2': 0.9
                    }, 10)
            ]
    else:
        if by_epoch:
            calls = [
                call('train', {
                    'learning_rate': 0.018000000000000002,
                    'momentum': 0.95
                }, 1),
                call('train', {
                    'learning_rate': 0.02,
                    'momentum': 0.95
                }, 11),
                call('train', {
                    'learning_rate': 0.018090169943749474,
                    'momentum': 0.95
                }, 61),
                call('train', {
                    'learning_rate': 0.0019098300562505265,
                    'momentum': 0.95
                }, 100)
            ]
        else:
            calls = [
                call('train', {
                    'learning_rate': 0.018000000000000002,
                    'momentum': 0.95
                }, 1),
                call('train', {
                    'learning_rate': 0.02,
                    'momentum': 0.95
                }, 6),
                call('train', {
                    'learning_rate': 0.018090169943749474,
                    'momentum': 0.95
                }, 7),
                call('train', {
                    'learning_rate': 0.0019098300562505265,
                    'momentum': 0.95
                }, 10)
            ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize('multi_optimziers, max_iters', [(True, 10), (True, 2),
                                                         (False, 10),
                                                         (False, 2)])
def test_one_cycle_runner_hook(multi_optimziers, max_iters):
    """Test OneCycleLrUpdaterHook and OneCycleMomentumUpdaterHook."""
    with pytest.raises(AssertionError):
        # by_epoch should be False
        OneCycleLrUpdaterHook(max_lr=0.1, by_epoch=True)

    with pytest.raises(ValueError):
        # expected float between 0 and 1
        OneCycleLrUpdaterHook(max_lr=0.1, pct_start=-0.1)

    with pytest.raises(ValueError):
        # anneal_strategy should be either 'cos' or 'linear'
        OneCycleLrUpdaterHook(max_lr=0.1, anneal_strategy='sin')

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='OneCycleMomentumUpdaterHook',
        base_momentum=0.85,
        max_momentum=0.95,
        pct_start=0.5,
        anneal_strategy='cos',
        three_phase=False)
    runner.register_hook_from_cfg(hook_cfg)

    # add LR scheduler
    hook_cfg = dict(
        type='OneCycleLrUpdaterHook',
        max_lr=0.01,
        pct_start=0.5,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
        three_phase=False)
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
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.0003999999999999993,
                    'learning_rate/model2': 0.0003999999999999993,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.95,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.00904508879153485,
                    'learning_rate/model2': 0.00904508879153485,
                    'momentum/model1': 0.8595491502812526,
                    'momentum/model2': 0.8595491502812526,
                }, 6),
            call(
                'train', {
                    'learning_rate/model1': 4e-08,
                    'learning_rate/model2': 4e-08,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.95,
                }, 10)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.0003999999999999993,
                'momentum': 0.95
            }, 1),
            call(
                'train', {
                    'learning_rate': 0.00904508879153485,
                    'momentum': 0.8595491502812526
                }, 6),
            call('train', {
                'learning_rate': 4e-08,
                'momentum': 0.95
            }, 10)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)

    # Test OneCycleLrUpdaterHook
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(
        runner_type='IterBasedRunner', max_epochs=None, max_iters=max_iters)

    args = dict(
        max_lr=0.01,
        total_steps=5,
        pct_start=0.5,
        anneal_strategy='linear',
        div_factor=25,
        final_div_factor=1e4,
    )
    hook = OneCycleLrUpdaterHook(**args)
    runner.register_hook(hook)
    if max_iters == 10:
        # test total_steps < max_iters
        with pytest.raises(ValueError):
            runner.run([loader], [('train', 1)])
    else:
        # test total_steps > max_iters
        runner.run([loader], [('train', 1)])
        lr_last = runner.current_lr()
        t = torch.tensor([0.0], requires_grad=True)
        optim = torch.optim.SGD([t], lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, **args)
        lr_target = []
        for _ in range(max_iters):
            optim.step()
            lr_target.append(optim.param_groups[0]['lr'])
            lr_scheduler.step()
        assert lr_target[-1] == lr_last[0]


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_cosine_restart_lr_update_hook(multi_optimziers):
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
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

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
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.01,
                    'learning_rate/model2': 0.005,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.01,
                    'learning_rate/model2': 0.005,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 6),
            call(
                'train', {
                    'learning_rate/model1': 0.0009549150281252633,
                    'learning_rate/model2': 0.00047745751406263163,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 10)
        ]
    else:
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


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_step_runner_hook(multi_optimziers):
    """Test StepLrUpdaterHook."""
    with pytest.raises(TypeError):
        # `step` should be specified
        StepLrUpdaterHook()
    with pytest.raises(AssertionError):
        # if `step` is int, should be positive
        StepLrUpdaterHook(-10)
    with pytest.raises(AssertionError):
        # if `step` is list of int, should all be positive
        StepLrUpdaterHook([10, 16, -20])

    # test StepLrUpdaterHook with int `step` value
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((30, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='StepMomentumUpdaterHook',
        by_epoch=False,
        step=5,
        gamma=0.5,
        min_momentum=0.05)
    runner.register_hook_from_cfg(hook_cfg)

    # add step LR scheduler
    hook = StepLrUpdaterHook(by_epoch=False, step=5, gamma=0.5, min_lr=1e-3)
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.01,
                    'learning_rate/model2': 0.005,
                    'momentum/model1': 0.475,
                    'momentum/model2': 0.45
                }, 6),
            call(
                'train', {
                    'learning_rate/model1': 0.0025,
                    'learning_rate/model2': 0.00125,
                    'momentum/model1': 0.11875,
                    'momentum/model2': 0.1125
                }, 16),
            call(
                'train', {
                    'learning_rate/model1': 0.00125,
                    'learning_rate/model2': 0.001,
                    'momentum/model1': 0.059375,
                    'momentum/model2': 0.05625
                }, 21),
            call(
                'train', {
                    'learning_rate/model1': 0.001,
                    'learning_rate/model2': 0.001,
                    'momentum/model1': 0.05,
                    'momentum/model2': 0.05
                }, 26),
            call(
                'train', {
                    'learning_rate/model1': 0.001,
                    'learning_rate/model2': 0.001,
                    'momentum/model1': 0.05,
                    'momentum/model2': 0.05
                }, 30)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.01,
                'momentum': 0.475
            }, 6),
            call('train', {
                'learning_rate': 0.0025,
                'momentum': 0.11875
            }, 16),
            call('train', {
                'learning_rate': 0.00125,
                'momentum': 0.059375
            }, 21),
            call('train', {
                'learning_rate': 0.001,
                'momentum': 0.05
            }, 26),
            call('train', {
                'learning_rate': 0.001,
                'momentum': 0.05
            }, 30)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)

    # test StepLrUpdaterHook with list[int] `step` value
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    # add momentum scheduler
    hook_cfg = dict(
        type='StepMomentumUpdaterHook',
        by_epoch=False,
        step=[4, 6, 8],
        gamma=0.1)
    runner.register_hook_from_cfg(hook_cfg)

    # add step LR scheduler
    hook = StepLrUpdaterHook(by_epoch=False, step=[4, 6, 8], gamma=0.1)
    runner.register_hook(hook)
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.002,
                    'learning_rate/model2': 0.001,
                    'momentum/model1': 9.5e-2,
                    'momentum/model2': 9.000000000000001e-2
                }, 5),
            call(
                'train', {
                    'learning_rate/model1': 2.0000000000000004e-4,
                    'learning_rate/model2': 1.0000000000000002e-4,
                    'momentum/model1': 9.500000000000001e-3,
                    'momentum/model2': 9.000000000000003e-3
                }, 7),
            call(
                'train', {
                    'learning_rate/model1': 2.0000000000000005e-05,
                    'learning_rate/model2': 1.0000000000000003e-05,
                    'momentum/model1': 9.500000000000002e-4,
                    'momentum/model2': 9.000000000000002e-4
                }, 9)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.002,
                'momentum': 0.095
            }, 5),
            call(
                'train', {
                    'learning_rate': 2.0000000000000004e-4,
                    'momentum': 9.500000000000001e-3
                }, 7),
            call(
                'train', {
                    'learning_rate': 2.0000000000000005e-05,
                    'momentum': 9.500000000000002e-4
                }, 9)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


@pytest.mark.parametrize('multi_optimizers, max_iters', [(True, 8),
                                                         (False, 8)])
def test_cyclic_lr_update_hook(multi_optimizers, max_iters):
    """Test CyclicLrUpdateHook."""
    with pytest.raises(AssertionError):
        # by_epoch should be False
        CyclicLrUpdaterHook(by_epoch=True)

    with pytest.raises(AssertionError):
        # target_ratio" must be either float or tuple/list of two floats
        CyclicLrUpdaterHook(by_epoch=False, target_ratio=(10.0, 0.1, 0.2))

    with pytest.raises(AssertionError):
        # step_ratio_up" must be in range [0,1)
        CyclicLrUpdaterHook(by_epoch=False, step_ratio_up=1.4)

    with pytest.raises(ValueError):
        # anneal_strategy must be one of "cos" or "linear"
        CyclicLrUpdaterHook(by_epoch=False, anneal_strategy='sin')

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(
        runner_type='IterBasedRunner',
        max_epochs=None,
        max_iters=max_iters,
        multi_optimziers=multi_optimizers)

    # add cyclic LR scheduler
    hook = CyclicLrUpdaterHook(
        by_epoch=False,
        target_ratio=(10.0, 1.0),
        cyclic_times=1,
        step_ratio_up=0.5,
        anneal_strategy='linear')
    runner.register_hook(hook)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())
    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    assert hasattr(hook, 'writer')
    if multi_optimizers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.02,
                    'learning_rate/model2': 0.01,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.155,
                    'learning_rate/model2': 0.0775,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 4),
            call(
                'train', {
                    'learning_rate/model1': 0.155,
                    'learning_rate/model2': 0.0775,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9,
                }, 6)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.02,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.155,
                'momentum': 0.95
            }, 4),
            call('train', {
                'learning_rate': 0.155,
                'momentum': 0.95
            }, 6),
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
    hook = WandbLoggerHook(log_artifact=True)
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
    hook.wandb.log_artifact.assert_called()
    hook.wandb.join.assert_called_with()


def test_neptune_hook():
    sys.modules['neptune'] = MagicMock()
    sys.modules['neptune.new'] = MagicMock()
    runner = _build_demo_runner()
    hook = NeptuneLoggerHook()

    loader = DataLoader(torch.ones((5, 2)))

    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    hook.neptune.init.assert_called_with()
    hook.run['momentum'].log.assert_called_with(0.95, step=6)
    hook.run.stop.assert_called_with()


def test_dvclive_hook():
    sys.modules['dvclive'] = MagicMock()
    runner = _build_demo_runner()

    hook = DvcliveLoggerHook()
    dvclive_mock = hook.dvclive
    loader = DataLoader(torch.ones((5, 2)))

    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)

    dvclive_mock.set_step.assert_called_with(6)
    dvclive_mock.log.assert_called_with('momentum', 0.95)


def test_dvclive_hook_model_file(tmp_path):
    sys.modules['dvclive'] = MagicMock()
    runner = _build_demo_runner()

    hook = DvcliveLoggerHook(model_file=osp.join(runner.work_dir, 'model.pth'))
    runner.register_hook(hook)

    loader = torch.utils.data.DataLoader(torch.ones((5, 2)))
    loader = DataLoader(torch.ones((5, 2)))

    runner.run([loader, loader], [('train', 1), ('val', 1)])

    assert osp.exists(osp.join(runner.work_dir, 'model.pth'))

    shutil.rmtree(runner.work_dir)


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimziers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimziers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

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
    return runner


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimziers=False):
    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimziers)

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


def test_get_triggered_stages():

    class ToyHook(Hook):
        # test normal stage
        def before_run():
            pass

        # test the method mapped to multi stages.
        def after_epoch():
            pass

    hook = ToyHook()
    # stages output have order, so here is list instead of set.
    expected_stages = ['before_run', 'after_train_epoch', 'after_val_epoch']
    assert hook.get_triggered_stages() == expected_stages


def test_gradient_cumulative_optimizer_hook():

    class ToyModel(nn.Module):

        def __init__(self, with_norm=False):
            super().__init__()
            self.fp16_enabled = False
            self.fc = nn.Linear(3, 2)
            nn.init.constant_(self.fc.weight, 1.)
            nn.init.constant_(self.fc.bias, 1.)
            self.with_norm = with_norm
            if with_norm:
                self.norm = nn.BatchNorm1d(2)

        def forward(self, x):
            x = self.fc(x)
            if self.with_norm:
                x = self.norm(x)
            return x

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

    def build_toy_runner(config=dict(type='EpochBasedRunner', max_epochs=3)):
        model = ToyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        tmp_dir = tempfile.mkdtemp()

        runner = build_runner(
            config,
            default_args=dict(
                model=model,
                work_dir=tmp_dir,
                optimizer=optimizer,
                logger=logging.getLogger(),
                meta=dict()))
        return runner

    with pytest.raises(AssertionError):
        # cumulative_iters only accepts int
        GradientCumulativeOptimizerHook(cumulative_iters='str')

    with pytest.raises(AssertionError):
        # cumulative_iters only accepts positive number
        GradientCumulativeOptimizerHook(cumulative_iters=-1)

    # test epoch based runner
    data = torch.rand((6, 3))
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner()
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2 = DataLoader(data, batch_size=3)
    runner_2 = build_toy_runner()
    optimizer_hook = OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2], [('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

    # test iter based runner
    data = torch.rand((8, 3))
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner(dict(type='IterBasedRunner', max_iters=8))
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2_divisible = DataLoader(data[:6], batch_size=3)
    loader_2_remainder = DataLoader(data[6:], batch_size=2)
    runner_2 = build_toy_runner(dict(type='IterBasedRunner', max_iters=3))
    optimizer_hook = OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2_divisible, loader_2_remainder], [('train', 2),
                                                            ('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

    # test has_batch_norm
    model = ToyModel(with_norm=True)
    optimizer_hook = GradientCumulativeOptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    assert optimizer_hook.has_batch_norm(model)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
def test_gradient_cumulative_fp16_optimizer_hook():

    class ToyModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.fp16_enabled = False
            self.fc = nn.Linear(3, 2)
            nn.init.constant_(self.fc.weight, 1.)
            nn.init.constant_(self.fc.bias, 1.)

        @auto_fp16(apply_to=('x', ))
        def forward(self, x):
            x = self.fc(x)
            return x

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

    def build_toy_runner(config=dict(type='EpochBasedRunner', max_epochs=3)):
        model = ToyModel().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        tmp_dir = tempfile.mkdtemp()

        runner = build_runner(
            config,
            default_args=dict(
                model=model,
                work_dir=tmp_dir,
                optimizer=optimizer,
                logger=logging.getLogger(),
                meta=dict()))
        return runner

    # test epoch based runner
    data = torch.rand((6, 3)).cuda()
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner()
    optimizer_hook = GradientCumulativeFp16OptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2 = DataLoader(data, batch_size=3)
    runner_2 = build_toy_runner()
    optimizer_hook = Fp16OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2], [('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)

    # test iter based runner
    data = torch.rand((8, 3)).cuda()
    # optimize with cumulative_iters
    loader_1 = DataLoader(data, batch_size=1)
    runner_1 = build_toy_runner(dict(type='IterBasedRunner', max_iters=8))
    optimizer_hook = GradientCumulativeFp16OptimizerHook(
        grad_clip=dict(max_norm=0.2), cumulative_iters=3)
    runner_1.register_hook(optimizer_hook)
    runner_1.run([loader_1], [('train', 1)])

    # optimize without cumulative_iters
    loader_2_divisible = DataLoader(data[:6], batch_size=3)
    loader_2_remainder = DataLoader(data[6:], batch_size=2)
    runner_2 = build_toy_runner(dict(type='IterBasedRunner', max_iters=3))
    optimizer_hook = Fp16OptimizerHook(grad_clip=dict(max_norm=0.2))
    runner_2.register_hook(optimizer_hook)
    runner_2.run([loader_2_divisible, loader_2_remainder], [('train', 2),
                                                            ('train', 1)])

    # test optimizer works well
    assert (runner_1.model.fc.weight < 1).all()
    assert (runner_1.model.fc.bias < 1).all()
    # test optimizer with cumulative_iters gets the same results
    assert torch.allclose(runner_1.model.fc.weight, runner_2.model.fc.weight)
    assert torch.allclose(runner_1.model.fc.bias, runner_2.model.fc.bias)
    shutil.rmtree(runner_1.work_dir)
    shutil.rmtree(runner_2.work_dir)
