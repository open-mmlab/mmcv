# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
import warnings


def test_save_checkpoint():
    try:
        import torch
        from torch import nn
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return

    import mmcv.runner

    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(
        model=model, batch_processor=lambda x: x, logger=logging.getLogger())

    with tempfile.TemporaryDirectory() as root:
        runner.save_checkpoint(root)

        latest_path = osp.join(root, 'latest.pth')
        epoch1_path = osp.join(root, 'epoch_1.pth')

        assert osp.exists(latest_path)
        assert osp.exists(epoch1_path)
        assert osp.realpath(latest_path) == osp.realpath(epoch1_path)

        torch.load(latest_path)


def test_build_lr_momentum_hook():
    try:
        from torch import nn
    except ImportError:
        warnings.warn('Skipping test_save_checkpoint in the absense of torch')
        return
    import mmcv.runner
    model = nn.Linear(1, 1)
    runner = mmcv.runner.Runner(
        model=model, batch_processor=lambda x: x, logger=logging.getLogger())

    # test policy that is already title
    lr_config = dict(
        policy='CosineAnealing',
        by_epoch=False,
        min_lr_ratio=0,
        warmup_iters=2,
        warmup_ratio=0.9)
    runner.register_lr_hook(lr_config)
    assert len(runner.hooks) == 1

    # test policy that is already title
    lr_config = dict(
        policy='Cyclic',
        by_epoch=False,
        target_ratio=(10, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_lr_hook(lr_config)
    assert len(runner.hooks) == 2

    # test policy that is not title
    lr_config = dict(
        policy='cyclic',
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_lr_hook(lr_config)
    assert len(runner.hooks) == 3

    # test policy that is title
    lr_config = dict(
        policy='Step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[8, 11])
    runner.register_lr_hook(lr_config)
    assert len(runner.hooks) == 4

    # test policy that is not title
    lr_config = dict(
        policy='step',
        warmup='linear',
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[8, 11])
    runner.register_lr_hook(lr_config)
    assert len(runner.hooks) == 5

    # test policy that is already title
    mom_config = dict(
        policy='CosineAnealing',
        min_momentum_ratio=0.99 / 0.95,
        by_epoch=False,
        warmup_iters=2,
        warmup_ratio=0.9 / 0.95)
    runner.register_momentum_hook(mom_config)
    assert len(runner.hooks) == 6

    # test policy that is already title
    mom_config = dict(
        policy='Cyclic',
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_momentum_hook(mom_config)
    assert len(runner.hooks) == 7

    # test policy that is already title
    mom_config = dict(
        policy='cyclic',
        by_epoch=False,
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4)
    runner.register_momentum_hook(mom_config)
    assert len(runner.hooks) == 8
