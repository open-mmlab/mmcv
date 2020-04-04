import os.path as osp
import sys
import warnings
from unittest.mock import MagicMock

import pytest

import mmcv.runner

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    warnings.warn('torch is not available')
    torch = None

only_if_torch_available = pytest.mark.skipif(
    torch is None, reason='torch is not available')


@only_if_torch_available
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


@only_if_torch_available
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
        experiment_name='test', log_model=log_model)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)

    hook.mlflow.set_experiment.assert_called_with('test')
    hook.mlflow.log_metrics.assert_called_with({'accuracy/val': 0.98}, step=5)
    if log_model:
        hook.mlflow_pytorch.log_model.assert_called_with(
            runner.model, 'models')
    else:
        assert not hook.mlflow_pytorch.log_model.called


@only_if_torch_available
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
