import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import CyclicLrUpdaterHook, IterTimerHook, PaviLoggerHook


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
