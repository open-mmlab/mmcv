import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import (FlatCosineAnnealingLrUpdaterHook, IterTimerHook,
                         PaviLoggerHook)


@pytest.mark.parametrize('multi_optimizers, by_epoch', [(False, False),
                                                        (True, False),
                                                        (False, True),
                                                        (True, True)])
def test_flat_cosine_runner_hook(multi_optimizers, by_epoch):
    """xdoctest -m tests/test_hooks.py test_flat_cosine_runner_hook."""
    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    max_epochs = 10 if by_epoch else 1
    runner = _build_demo_runner(
        multi_optimizers=multi_optimizers, max_epochs=max_epochs)

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
    if multi_optimizers:
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
