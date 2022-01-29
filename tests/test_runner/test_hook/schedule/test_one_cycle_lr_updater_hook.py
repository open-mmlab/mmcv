import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import IterTimerHook, OneCycleLrUpdaterHook, PaviLoggerHook


@pytest.mark.parametrize('multi_optimizers, max_iters', [(True, 10), (True, 2),
                                                         (False, 10),
                                                         (False, 2)])
def test_one_cycle_runner_hook(multi_optimizers, max_iters):
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
