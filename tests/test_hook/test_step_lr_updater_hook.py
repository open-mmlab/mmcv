import shutil
import sys
from unittest.mock import MagicMock, call

import pytest
import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner import IterTimerHook, PaviLoggerHook, StepLrUpdaterHook


@pytest.mark.parametrize('multi_optimizers', (True, False))
def test_step_runner_hook(multi_optimizers):
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
    runner = _build_demo_runner(multi_optimizers=multi_optimizers)

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
    if multi_optimizers:
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
