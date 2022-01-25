import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import (CosineRestartLrUpdaterHook, IterTimerHook,
                         PaviLoggerHook)


@pytest.mark.parametrize('multi_optimizers', (True, False))
def test_cosine_restart_lr_update_hook(multi_optimizers):
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
