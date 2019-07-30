import os.path as osp
import tempfile
import warnings


def test_save_checkpoint():
    try:
        import torch
        import torch.nn as nn
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
        assert osp.realpath(latest_path) == epoch1_path

        torch.load(latest_path)
