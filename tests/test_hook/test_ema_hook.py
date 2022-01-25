import shutil

import torch
import torch.nn as nn
from tests.test_hook.test_utils import _build_demo_runner
from torch.nn.init import constant_
from torch.utils.data import DataLoader

from mmcv.runner import CheckpointHook, EMAHook


def test_ema_hook():
    """xdoctest -m tests/test_hooks.py test_ema_hook."""

    class DemoModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=1,
                padding=1,
                bias=True)
            self._init_weight()

        def _init_weight(self):
            constant_(self.conv.weight, 0)
            constant_(self.conv.bias, 0)

        def forward(self, x):
            return self.conv(x).sum()

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    loader = DataLoader(torch.ones((1, 1, 1, 1)))
    runner = _build_demo_runner()
    demo_model = DemoModel()
    runner.model = demo_model
    emahook = EMAHook(momentum=0.1, interval=2, warm_up=100, resume_from=None)
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(emahook, priority='HIGHEST')
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_1.pth')
    contain_ema_buffer = False
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            contain_ema_buffer = True
            assert value.sum() == 0
            value.fill_(1)
        else:
            assert value.sum() == 0
    assert contain_ema_buffer
    torch.save(checkpoint, f'{runner.work_dir}/epoch_1.pth')
    work_dir = runner.work_dir
    resume_ema_hook = EMAHook(
        momentum=0.5, warm_up=0, resume_from=f'{work_dir}/epoch_1.pth')
    runner = _build_demo_runner(max_epochs=2)
    runner.model = demo_model
    runner.register_hook(resume_ema_hook, priority='HIGHEST')
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_2.pth')
    contain_ema_buffer = False
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            contain_ema_buffer = True
            assert value.sum() == 2
        else:
            assert value.sum() == 1
    assert contain_ema_buffer
    shutil.rmtree(runner.work_dir)
    shutil.rmtree(work_dir)
