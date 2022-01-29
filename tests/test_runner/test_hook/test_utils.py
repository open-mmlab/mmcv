import logging
import sys
import tempfile
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from mmcv.runner import build_runner

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimizers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimizers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimizers=False):
    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimizers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner
