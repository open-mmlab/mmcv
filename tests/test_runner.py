# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import sys
import tempfile
import warnings
from unittest.mock import MagicMock

sys.modules['wandb'] = MagicMock()


def test_save_checkpoint():
    try:
        import torch
        from torch import nn
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return

    import mmcv.runner

    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(model=model, batch_processor=lambda x: x)

    with tempfile.TemporaryDirectory() as root:
        runner.save_checkpoint(root)

        latest_path = osp.join(root, 'latest.pth')
        epoch1_path = osp.join(root, 'epoch_1.pth')

        assert osp.exists(latest_path)
        assert osp.exists(epoch1_path)
        assert osp.realpath(latest_path) == osp.realpath(epoch1_path)

        torch.load(latest_path)


def test_wandb_hook():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return

    import mmcv.runner
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
