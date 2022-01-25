import os.path as osp
import platform
import shutil
import sys
from unittest.mock import MagicMock

import torch
from tests.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.runner.hooks import PaviLoggerHook


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
