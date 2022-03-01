# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import random
import string
import tempfile

import pytest
import torch.nn as nn

from mmcv.runner import build_runner

# Most of its functions are inherited from EpochBasedRunner and IterBasedRunner
# So only do incremental testing on overridden methods
# Comparing with base runner,
# Overridden functions are listed below:
# __init__, register_lr_hook, register_optimizer_hook
# register_lr_hook and register_optimizer_hook are tested in test_runner.py


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
    # __init__
    temp_root = tempfile.gettempdir()
    dir_name = ''.join(
        [random.choice(string.ascii_letters) for _ in range(10)])

    default_args = dict(
        model=Model(),
        work_dir=osp.join(temp_root, dir_name),
        logger=logging.getLogger())
    cfg = dict(type='IpuEpochBasedRunner', max_epochs=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_epochs == 1
    cfg = dict(type='IpuIterBasedRunner', max_iters=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_iters == 1

    with pytest.raises(ValueError, match='Only one of'):
        cfg = dict(type='IpuIterBasedRunner', max_epochs=1, max_iters=1)
        runner = build_runner(cfg, default_args=default_args)
