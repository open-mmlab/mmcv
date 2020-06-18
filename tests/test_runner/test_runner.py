# Copyright (c) Open-MMLab. All rights reserved.
import logging
import os
import os.path as osp
import random
import string
import tempfile

import pytest
import torch
import torch.nn as nn

from mmcv.parallel import MMDataParallel
from mmcv.runner import EpochBasedRunner


class OldStyleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)


class Model(OldStyleModel):

    def train_step(self):
        pass

    def val_step(self):
        pass


def test_epoch_based_runner():

    with pytest.warns(UserWarning):
        # batch_processor is deprecated
        model = OldStyleModel()

        def batch_processor():
            pass

        _ = EpochBasedRunner(
            model, batch_processor, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # batch_processor must be callable
        model = OldStyleModel()
        _ = EpochBasedRunner(
            model, batch_processor=0, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # optimizer must be a optimizer or a dict of optimizers
        model = Model()
        optimizer = 'NotAOptimizer'
        _ = EpochBasedRunner(
            model, optimizer=optimizer, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # optimizer must be a optimizer or a dict of optimizers
        model = Model()
        optimizers = dict(optim1=torch.optim.Adam(), optim2='NotAOptimizer')
        _ = EpochBasedRunner(
            model, optimizer=optimizers, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # logger must be a logging.Logger
        model = Model()
        _ = EpochBasedRunner(model, logger=None)

    with pytest.raises(TypeError):
        # meta must be a dict or None
        model = Model()
        _ = EpochBasedRunner(model, logger=logging.getLogger(), meta=['list'])

    with pytest.raises(AssertionError):
        # model must implement the method train_step()
        model = OldStyleModel()
        _ = EpochBasedRunner(model, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # work_dir must be a str or None
        model = Model()
        _ = EpochBasedRunner(model, work_dir=1, logger=logging.getLogger())

    with pytest.raises(RuntimeError):
        # batch_processor and train_step() cannot be both set

        def batch_processor():
            pass

        model = Model()
        _ = EpochBasedRunner(
            model, batch_processor, logger=logging.getLogger())

    # test work_dir
    model = Model()
    temp_root = tempfile.gettempdir()
    dir_name = ''.join(
        [random.choice(string.ascii_letters) for _ in range(10)])
    work_dir = osp.join(temp_root, dir_name)
    _ = EpochBasedRunner(model, work_dir=work_dir, logger=logging.getLogger())
    assert osp.isdir(work_dir)
    _ = EpochBasedRunner(model, work_dir=work_dir, logger=logging.getLogger())
    assert osp.isdir(work_dir)
    os.removedirs(work_dir)


def test_runner_with_parallel():

    def batch_processor():
        pass

    model = MMDataParallel(OldStyleModel())
    _ = EpochBasedRunner(model, batch_processor, logger=logging.getLogger())

    model = MMDataParallel(Model())
    _ = EpochBasedRunner(model, logger=logging.getLogger())

    with pytest.raises(RuntimeError):
        # batch_processor and train_step() cannot be both set

        def batch_processor():
            pass

        model = MMDataParallel(Model())
        _ = EpochBasedRunner(
            model, batch_processor, logger=logging.getLogger())


def test_save_checkpoint():
    model = Model()
    runner = EpochBasedRunner(model=model, logger=logging.getLogger())

    with tempfile.TemporaryDirectory() as root:
        runner.save_checkpoint(root)

        latest_path = osp.join(root, 'latest.pth')
        epoch1_path = osp.join(root, 'epoch_1.pth')

        assert osp.exists(latest_path)
        assert osp.exists(epoch1_path)
        assert osp.realpath(latest_path) == osp.realpath(epoch1_path)

        torch.load(latest_path)


def test_build_lr_momentum_hook():
    model = Model()
    runner = EpochBasedRunner(model=model, logger=logging.getLogger())

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
