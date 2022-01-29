import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import IterTimerHook, PaviLoggerHook


@pytest.mark.parametrize('multi_optimizers', (True, False))
def test_cosine_runner_hook(multi_optimizers):
    """xdoctest -m tests/test_hooks.py test_cosine_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
