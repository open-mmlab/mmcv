import os.path as osp
import sys
import warnings

from mock import MagicMock

import mmcv.runner


def test_pavi_hook():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn('Skipping test_pavi_hook in the absense of torch')
        return
    sys.modules['pavi'] = MagicMock()

    model = nn.Linear(1, 1)
    loader = DataLoader(torch.ones((5, 5)))
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    runner = mmcv.runner.Runner(
        model=model,
        work_dir=work_dir,
        batch_processor=lambda model, x, **kwargs: {
            'log_vars': {
                'loss': 2.333
            },
            'num_samples': 5
        })

    hook = mmcv.runner.hooks.PaviLoggerHook(
        add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader, loader], [('train', 1), ('val', 1)], 1)

    assert hasattr(hook, 'writer')
    hook.writer.add_scalars.assert_called_with('val', {'loss': 2.333}, 5)
    hook.writer.add_snapshot_file.assert_called_with(
        tag='data',
        snapshot_file_path=osp.join(work_dir, 'latest.pth'),
        iteration=5)
