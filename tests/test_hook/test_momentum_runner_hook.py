import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import PaviLoggerHook


@pytest.mark.parametrize('multi_optimizers, max_iters, gamma, cyclic_times',
                         [(True, 8, 1, 1), (False, 8, 0.5, 2)])
def test_momentum_runner_hook(multi_optimizers, max_iters, gamma,
                              cyclic_times):
    """xdoctest -m tests/test_hooks.py test_momentum_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

    # add momentum scheduler
    hook_cfg = dict(
        type='CyclicMomentumUpdaterHook',
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=cyclic_times,
        step_ratio_up=0.4,
        gamma=gamma)
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
    if multi_optimizers:
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
                'learning_rate': 0.11,
                'momentum': 0.85
            }, 3),
            call('train', {
                'learning_rate': 0.1879422863405995,
                'momentum': 0.95
            }, 6),
            call('train', {
                'learning_rate': 0.11000000000000001,
                'momentum': 0.9
            }, 8),
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)

    # test constant momentum warmup
    sys.modules['pavi'] = MagicMock()
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
