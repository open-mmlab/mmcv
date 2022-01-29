import shutil
import sys
from unittest.mock import MagicMock

import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner.hooks import WandbLoggerHook


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
