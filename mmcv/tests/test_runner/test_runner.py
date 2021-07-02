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
from mmcv.runner import (RUNNERS, EpochBasedRunner, IterBasedRunner,
                         build_runner)
from mmcv.runner.hooks import IterTimerHook


class OldStyleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)


class Model(OldStyleModel):

    def train_step(self):
        pass

    def val_step(self):
        pass


def test_build_runner():
    temp_root = tempfile.gettempdir()
    dir_name = ''.join(
        [random.choice(string.ascii_letters) for _ in range(10)])

    default_args = dict(
        model=Model(),
        work_dir=osp.join(temp_root, dir_name),
        logger=logging.getLogger())
    cfg = dict(type='EpochBasedRunner', max_epochs=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_epochs == 1
    cfg = dict(type='IterBasedRunner', max_iters=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_iters == 1

    with pytest.raises(ValueError, match='Only one of'):
        cfg = dict(type='IterBasedRunner', max_epochs=1, max_iters=1)
        runner = build_runner(cfg, default_args=default_args)


@pytest.mark.parametrize('runner_class', RUNNERS.module_dict.values())
def test_epoch_based_runner(runner_class):

    with pytest.warns(UserWarning):
        # batch_processor is deprecated
        model = OldStyleModel()

        def batch_processor():
            pass

        _ = runner_class(model, batch_processor, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # batch_processor must be callable
        model = OldStyleModel()
        _ = runner_class(model, batch_processor=0, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # optimizer must be a optimizer or a dict of optimizers
        model = Model()
        optimizer = 'NotAOptimizer'
        _ = runner_class(
            model, optimizer=optimizer, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # optimizer must be a optimizer or a dict of optimizers
        model = Model()
        optimizers = dict(optim1=torch.optim.Adam(), optim2='NotAOptimizer')
        _ = runner_class(
            model, optimizer=optimizers, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # logger must be a logging.Logger
        model = Model()
        _ = runner_class(model, logger=None)

    with pytest.raises(TypeError):
        # meta must be a dict or None
        model = Model()
        _ = runner_class(model, logger=logging.getLogger(), meta=['list'])

    with pytest.raises(AssertionError):
        # model must implement the method train_step()
        model = OldStyleModel()
        _ = runner_class(model, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # work_dir must be a str or None
        model = Model()
        _ = runner_class(model, work_dir=1, logger=logging.getLogger())

    with pytest.raises(RuntimeError):
        # batch_processor and train_step() cannot be both set

        def batch_processor():
            pass

        model = Model()
        _ = runner_class(model, batch_processor, logger=logging.getLogger())

    # test work_dir
    model = Model()
    temp_root = tempfile.gettempdir()
    dir_name = ''.join(
        [random.choice(string.ascii_letters) for _ in range(10)])
    work_dir = osp.join(temp_root, dir_name)
    _ = runner_class(model, work_dir=work_dir, logger=logging.getLogger())
    assert osp.isdir(work_dir)
    _ = runner_class(model, work_dir=work_dir, logger=logging.getLogger())
    assert osp.isdir(work_dir)
    os.removedirs(work_dir)


@pytest.mark.parametrize('runner_class', RUNNERS.module_dict.values())
def test_runner_with_parallel(runner_class):

    def batch_processor():
        pass

    model = MMDataParallel(OldStyleModel())
    _ = runner_class(model, batch_processor, logger=logging.getLogger())

    model = MMDataParallel(Model())
    _ = runner_class(model, logger=logging.getLogger())

    with pytest.raises(RuntimeError):
        # batch_processor and train_step() cannot be both set

        def batch_processor():
            pass

        model = MMDataParallel(Model())
        _ = runner_class(model, batch_processor, logger=logging.getLogger())


@pytest.mark.parametrize('runner_class', RUNNERS.module_dict.values())
def test_save_checkpoint(runner_class):
    model = Model()
    runner = runner_class(model=model, logger=logging.getLogger())

    with pytest.raises(TypeError):
        # meta should be None or dict
        runner.save_checkpoint('.', meta=list())

    with tempfile.TemporaryDirectory() as root:
        runner.save_checkpoint(root)

        latest_path = osp.join(root, 'latest.pth')
        assert osp.exists(latest_path)

        if isinstance(runner, EpochBasedRunner):
            first_ckp_path = osp.join(root, 'epoch_1.pth')
        elif isinstance(runner, IterBasedRunner):
            first_ckp_path = osp.join(root, 'iter_1.pth')

        assert osp.exists(first_ckp_path)
        assert osp.realpath(latest_path) == osp.realpath(first_ckp_path)

        torch.load(latest_path)


@pytest.mark.parametrize('runner_class', RUNNERS.module_dict.values())
def test_build_lr_momentum_hook(runner_class):
    model = Model()
    runner = runner_class(model=model, logger=logging.getLogger())

    # test policy that is already title
    lr_config = dict(
        policy='CosineAnnealing',
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
        policy='CosineAnnealing',
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


@pytest.mark.parametrize('runner_class', RUNNERS.module_dict.values())
def test_register_timer_hook(runner_class):
    model = Model()
    runner = runner_class(model=model, logger=logging.getLogger())

    # test register None
    timer_config = None
    runner.register_timer_hook(timer_config)
    assert len(runner.hooks) == 0

    # test register IterTimerHook with config
    timer_config = dict(type='IterTimerHook')
    runner.register_timer_hook(timer_config)
    assert len(runner.hooks) == 1
    assert isinstance(runner.hooks[0], IterTimerHook)

    # test register IterTimerHook
    timer_config = IterTimerHook()
    runner.register_timer_hook(timer_config)
    assert len(runner.hooks) == 2
    assert isinstance(runner.hooks[1], IterTimerHook)
