import shutil

import torch
from torch.utils.data import DataLoader

from .test_utils import _build_demo_runner


def test_sync_buffers_hook():
    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.register_hook_from_cfg(dict(type='SyncBuffersHook'))
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)
