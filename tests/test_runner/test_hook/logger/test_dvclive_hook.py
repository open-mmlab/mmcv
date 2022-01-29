import os.path as osp
import shutil
import sys
from unittest.mock import MagicMock

import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner.hooks import DvcliveLoggerHook


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
