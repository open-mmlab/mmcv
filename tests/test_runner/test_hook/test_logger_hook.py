import os.path as osp
import platform
import shutil
import sys
from unittest.mock import MagicMock

import pytest
import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner.hooks import (DvcliveLoggerHook, MlflowLoggerHook,
                               NeptuneLoggerHook, PaviLoggerHook,
                               WandbLoggerHook)


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

    loader = DataLoader(torch.ones((5, 2)))

    runner.run([loader, loader], [('train', 1), ('val', 1)])

    assert osp.exists(osp.join(runner.work_dir, 'model.pth'))

    shutil.rmtree(runner.work_dir)


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
