import os.path as osp
import sys
import warnings
from unittest.mock import MagicMock

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


def test_momentum_runner_hook():
    """
    pytest tests/test_hooks.py -m test_runner_hook
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        warnings.warn('Skipping test_momentum_runner_hook in the absense of torch')
        return

    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner()

    # add momentum scheduler
    hook = mmcv.runner.hooks.momentum_updater.CyclicMomentumUpdaterHook(
        by_epoch=False,
        target_ratio=[0.85 / 0.95, 1],
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)

    # add momentum LR scheduler
    hook = mmcv.runner.hooks.lr_updater.CyclicLrUpdaterHook(
        by_epoch=False,
        target_ratio=[10, 1],
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_hook(hook)
    runner.register_hook(mmcv.runner.hooks.IterTimerHook())

    runner.run([loader], [('train', 1)], 1)
    log_path = osp.join(runner.work_dir, '{}.log.json'.format(runner.timestamp))
    import json
    with open(log_path, 'r') as f:
        log_jsons = f.readlines()
    log_jsons = [json.loads(l) for l in log_jsons]

    assert log_jsons[0]['momentum'] == 0.95
    assert log_jsons[4]['momentum'] == 0.85
    assert log_jsons[7]['momentum'] == 0.9
    assert log_jsons[0]['lr'] == 0.02
    assert log_jsons[4]['lr'] == 0.2
    assert log_jsons[7]['lr'] == 0.11


def _build_demo_runner():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 1)
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    log_config = dict(
        interval=1,
        hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = mmcv.runner.Runner(
        model=model,
        work_dir=work_dir,
        batch_processor=lambda model, x, **kwargs: {'loss': model(x)-0},
        optimizer=optimizer)

    runner.register_logger_hooks(log_config)
    return runner
