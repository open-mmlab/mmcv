import os.path as osp
import shutil
import sys
from unittest.mock import MagicMock, patch

import torch
from tests.test_runner.test_hook.test_utils import _build_demo_runner
from torch.utils.data import DataLoader

from mmcv.fileio.file_client import PetrelBackend
from mmcv.runner import CheckpointHook


def test_checkpoint_hook(tmp_path):
    """xdoctest -m tests/test_runner/test_hooks.py test_checkpoint_hook."""
    sys.modules['petrel_client'] = MagicMock()
    sys.modules['petrel_client.client'] = MagicMock()
    # test epoch based runner
    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner('EpochBasedRunner', max_epochs=1)
    runner.meta = dict()
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(checkpointhook)
    runner.run([loader], [('train', 1)])
    assert runner.meta['hook_msgs']['last_ckpt'] == osp.join(
        runner.work_dir, 'epoch_1.pth')
    shutil.rmtree(runner.work_dir)

    # test petrel oss when the type of runner is `EpochBasedRunner`
    runner = _build_demo_runner('EpochBasedRunner', max_epochs=4)
    runner.meta = dict()
    out_dir = 's3://user/data'
    with patch.object(PetrelBackend, 'put') as mock_put, \
            patch.object(PetrelBackend, 'remove') as mock_remove, \
            patch.object(PetrelBackend, 'isfile') as mock_isfile:
        checkpointhook = CheckpointHook(
            interval=1, out_dir=out_dir, by_epoch=True, max_keep_ckpts=2)
        runner.register_hook(checkpointhook)
        runner.run([loader], [('train', 1)])
        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        assert runner.meta['hook_msgs']['last_ckpt'] == \
               '/'.join([out_dir, basename, 'epoch_4.pth'])
    mock_put.assert_called()
    mock_remove.assert_called()
    mock_isfile.assert_called()
    shutil.rmtree(runner.work_dir)

    # test iter based runner
    runner = _build_demo_runner(
        'IterBasedRunner', max_iters=1, max_epochs=None)
    runner.meta = dict()
    checkpointhook = CheckpointHook(interval=1, by_epoch=False)
    runner.register_hook(checkpointhook)
    runner.run([loader], [('train', 1)])
    assert runner.meta['hook_msgs']['last_ckpt'] == osp.join(
        runner.work_dir, 'iter_1.pth')
    shutil.rmtree(runner.work_dir)

    # test petrel oss when the type of runner is `IterBasedRunner`
    runner = _build_demo_runner(
        'IterBasedRunner', max_iters=4, max_epochs=None)
    runner.meta = dict()
    out_dir = 's3://user/data'
    with patch.object(PetrelBackend, 'put') as mock_put, \
            patch.object(PetrelBackend, 'remove') as mock_remove, \
            patch.object(PetrelBackend, 'isfile') as mock_isfile:
        checkpointhook = CheckpointHook(
            interval=1, out_dir=out_dir, by_epoch=False, max_keep_ckpts=2)
        runner.register_hook(checkpointhook)
        runner.run([loader], [('train', 1)])
        basename = osp.basename(runner.work_dir.rstrip(osp.sep))
        assert runner.meta['hook_msgs']['last_ckpt'] == \
               '/'.join([out_dir, basename, 'iter_4.pth'])
    mock_put.assert_called()
    mock_remove.assert_called()
    mock_isfile.assert_called()
    shutil.rmtree(runner.work_dir)
