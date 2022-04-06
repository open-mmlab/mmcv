# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import random
import string
import tempfile

import pytest
import torch
import torch.nn as nn

from mmcv.runner import build_runner
from mmcv.utils.ipu_wrapper import IPU_MODE
from torch.utils.data import Dataset
if IPU_MODE:
    from mmcv.runner import ipu_runner, IPUEpochBasedRunner
    from mmcv.runner.ipu import IPUDataloader

skip_no_ipu = pytest.mark.skipif(
    not IPU_MODE, reason='test case under ipu environment')

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


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU6()
        self.fp16_enabled = False

    def forward(self, img, return_loss=True, **kwargs):
        x = self.conv(img)
        x = self.bn(x)
        x = self.relu(x)
        if return_loss:
            loss = ((x - kwargs['gt_label'])**2).sum()
            return {'loss': loss, 'loss1': loss+1}
        return x

    def _parse_losses(self, losses):
        return losses['loss'], {'loss1': losses['loss']}

    def train_step(self, data, optimizer=None, **kwargs):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs


class ToyDataset(Dataset):
    def __getitem__(self, index):
        return {'img': torch.rand((3, 10, 10)),
                'gt_label': torch.rand((3, 10, 10))}
    
    def __len__(self,):
        return 3


@skip_no_ipu
def test_build_runner():
    # __init__
    temp_root = tempfile.gettempdir()
    dir_name = ''.join(
        [random.choice(string.ascii_letters) for _ in range(10)])

    default_args = dict(
        model=Model(),
        work_dir=osp.join(temp_root, dir_name),
        logger=logging.getLogger())
    cfg = dict(type='IPUEpochBasedRunner', max_epochs=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_epochs == 1
    cfg = dict(type='IPUIterBasedRunner', max_iters=1)
    runner = build_runner(cfg, default_args=default_args)
    assert runner._max_iters == 1

    ipu_runner.IPU_MODE = False
    cfg = dict(type='IPUIterBasedRunner', max_iters=1)
    with pytest.raises(
            NotImplementedError,
            match='cpu mode on IPURunner not supported'):
        runner = build_runner(cfg, default_args=default_args)

    ipu_runner.IPU_MODE = True
    with pytest.raises(ValueError, match='Only one of'):
        cfg = dict(type='IPUIterBasedRunner', max_epochs=1, max_iters=1)
        runner = build_runner(cfg, default_args=default_args)

    model = ToyModel()
    ipu_options = {'train_cfgs': {}, 'eval_cfgs': {}}
    dataloader = IPUDataloader(None, ToyDataset(), num_workers=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = dict(type='IPUIterBasedRunner',
               max_iters=2,
               ipu_options=ipu_options)
    default_args = dict(
        model=model,
        optimizer=optimizer,
        work_dir=osp.join(temp_root, dir_name),
        logger=logging.getLogger())
    runner = build_runner(cfg, default_args=default_args)
    runner.run([dataloader], [('train', 2)])
    runner.get_ipu_options('val')
    with pytest.raises(ValueError, match='mode should be train or val'):
        runner.get_ipu_options('666')
