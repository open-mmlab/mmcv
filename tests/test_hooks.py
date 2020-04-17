import os.path as osp
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv.runner


def test_pavi_hook():
    sys.modules['pavi'] = MagicMock()

    model = nn.Linear(1, 1)
    loader = DataLoader(torch.ones((5, 5)))
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    runner = mmcv.runner.Runner(
        model=model,
        work_dir=work_dir,
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                'loss': 2.333
            },
            'num_samples': 5
        })

    hook = mmcv.runner.hooks.PaviLoggerHook(
        add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)

    assert hasattr(hook, 'writer')
    hook.writer.add_scalars.assert_called_with('val', {'loss': 2.333}, 5)
    hook.writer.add_snapshot_file.assert_called_with(
        tag='data',
        snapshot_file_path=osp.join(work_dir, 'latest.pth'),
        iteration=5)


def test_momentum_runner_hook():
    """
    xdoctest -m tests/test_hooks.py test_momentum_runner_hook
    """

    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_model_runner()

    # add momentum scheduler
    hook = mmcv.runner.hooks.momentum_updater.CyclicMomentumUpdaterHook(
        by_epoch=False,
        target_ratio=[0.85 / 0.95, 1],
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)

    # add momentum LR scheduler
    hook = mmcv.runner.hooks.lr_updater.CyclicLrUpdaterHook(
        by_epoch=False,
        target_ratio=[10, 1],
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)
    runner.register_hook(mmcv.runner.hooks.IterTimerHook())

    runner.run([loader], [('train', 1)], 1)
    log_path = osp.join(runner.work_dir,
                        '{}.log.json'.format(runner.timestamp))
    import json
    with open(log_path, 'r') as f:
        log_jsons = f.readlines()
    log_jsons = [json.loads(l) for l in log_jsons]

    assert log_jsons[0]['momentum'] == 0.95
    assert log_jsons[4]['momentum'] == 0.85
    assert log_jsons[7]['momentum'] == 0.9
    assert log_jsons[0]['lr'] == 0.02
    assert log_jsons[4]['lr'] == 0.2
    assert log_jsons[7]['lr'] == 0.11


def test_cosine_runner_hook():
    """
    xdoctest -m tests/test_hooks.py test_cosine_runner_hook
    """

    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_model_runner()

    # add momentum scheduler
    hook = mmcv.runner.hooks.momentum_updater.CosineMomentumUpdaterHook(
        target=0.99 / 0.95,
        by_epoch=False,
        as_ratio=True,
        warmup_iters=2,
        warmup_ratio=0.9 / 0.95)
    runner.register_hook(hook)

    # add momentum LR scheduler
    hook = mmcv.runner.hooks.lr_updater.CosineLrUpdaterHook(
        target=0,
        by_epoch=False,
        as_ratio=True,
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_hook(hook)
    runner.register_hook(mmcv.runner.hooks.IterTimerHook())

    runner.run([loader], [('train', 1)], 1)
    log_path = osp.join(runner.work_dir,
                        '{}.log.json'.format(runner.timestamp))
    import json
    with open(log_path, 'r') as f:
        log_jsons = f.readlines()
    log_jsons = [json.loads(l) for l in log_jsons]
    assert log_jsons[0]['momentum'] == 0.95
    assert log_jsons[5]['momentum'] == 0.97
    assert log_jsons[9]['momentum'] == 0.98902
    assert log_jsons[0]['lr'] == 0.02
    assert log_jsons[5]['lr'] == 0.01
    assert log_jsons[9]['lr'] == 0.00049


@pytest.mark.parametrize('log_model', (True, False))
def test_mlflow_hook(log_model):
    sys.modules['mlflow'] = MagicMock()
    sys.modules['mlflow.pytorch'] = MagicMock()

    model = nn.Linear(1, 1)
    loader = DataLoader(torch.ones((5, 5)))
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    runner = mmcv.runner.Runner(
        model=model,
        work_dir=work_dir,
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 5
        })

    hook = mmcv.runner.hooks.MlflowLoggerHook(
        exp_name='test', log_model=log_model)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)

    hook.mlflow.set_experiment.assert_called_with('test')
    hook.mlflow.log_metrics.assert_called_with({'accuracy/val': 0.98}, step=5)
    if log_model:
        hook.mlflow_pytorch.log_model.assert_called_with(
            runner.model, 'models')
    else:
        assert not hook.mlflow_pytorch.log_model.called


def test_wandb_hook():
    sys.modules['wandb'] = MagicMock()

    hook = mmcv.runner.hooks.WandbLoggerHook()
    loader = DataLoader(torch.ones((5, 5)))

    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(
        model=model,
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                'accuracy': 0.98
            },
            'num_samples': 5
        })
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)
    hook.wandb.init.assert_called_with()
    hook.wandb.log.assert_called_with({'accuracy/val': 0.98}, step=5)
    hook.wandb.join.assert_called_with()


def _build_model_runner():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 1)
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = mmcv.runner.Runner(
        model=model,
        work_dir=work_dir,
        batch_processor=lambda model, x, **kwargs: {'loss': model(x) - 0},
        optimizer=optimizer)

    runner.register_logger_hooks(log_config)
    return runner
