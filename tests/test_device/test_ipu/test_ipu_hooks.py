# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp

import pytest
import torch
import torch.nn as nn

from mmcv.runner import build_runner
from mmcv.runner.fp16_utils import auto_fp16
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from mmcv.device.ipu.hook_wrapper import IPUFp16OptimizerHook

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU_AVAILABLE, reason='test case under ipu environment')


# TODO Once the model training and inference interfaces
# of MMCLS and MMDET are unified,
# construct the model according to the unified standards
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU6()
        self.fp16_enabled = False

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        x = self.conv(img)
        x = self.bn(x)
        x = self.relu(x)
        if return_loss:
            loss = ((x - kwargs['gt_label'])**2).sum()
            return {
                'loss': loss,
                'loss_list': [loss, loss],
                'loss_dict': {
                    'loss1': loss
                }
            }
        return x

    def _parse_losses(self, losses):
        return losses['loss'], losses['loss']

    def train_step(self, data, optimizer=None, **kwargs):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs


@skip_no_ipu
def test_ipu_hook_wrapper(tmp_path):

    model = ToyModel()
    dummy_input = {
        'data': {
            'img': torch.rand((16, 3, 10, 10)),
            'gt_label': torch.rand((16, 3, 10, 10))
        }
    }

    dir_name = 'a_tmp_dir'
    working_dir = osp.join(tmp_path, dir_name)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    default_args = dict(
        model=model,
        work_dir=working_dir,
        optimizer=optimizer,
        logger=logging.getLogger())
    cfg = dict(type='IPUEpochBasedRunner', max_epochs=1)
    dummy_runner = build_runner(cfg, default_args=default_args)

    # learning policy
    lr_config = dict(policy='step', step=[1, 150])
    # test optimizer config
    optimizer_config = dict(
        grad_clip=dict(max_norm=2), detect_anomalous_params=True)

    # test building ipu_lr_hook_class
    dummy_runner.register_training_hooks(
        lr_config=lr_config, optimizer_config=None, timer_config=None)

    # test _set_lr()
    output = dummy_runner.model.train_step(**dummy_input)
    dummy_runner.outputs = output
    dummy_runner.call_hook('before_train_epoch')

    # test building ipu_optimizer_hook_class
    with pytest.raises(
            NotImplementedError, match='IPU does not support gradient clip'):
        dummy_runner.register_training_hooks(
            lr_config=None,
            optimizer_config=optimizer_config,
            timer_config=None)

    # test fp16 optimizer hook
    lr_config = dict(policy='step', step=[1, 150])
    optimizer_config = dict(grad_clip=dict(max_norm=2))
    dummy_runner.hooks.pop(0)

    with pytest.raises(NotImplementedError, match='IPU mode does not support'):
        optimizer_config = IPUFp16OptimizerHook(
            loss_scale='dynamic', distributed=False)

    with pytest.raises(NotImplementedError, match='IPU mode supports single'):
        optimizer_config = IPUFp16OptimizerHook(
            loss_scale={}, distributed=False)

    with pytest.raises(ValueError, match='loss_scale should be float'):
        optimizer_config = IPUFp16OptimizerHook(
            loss_scale=[], distributed=False)

    optimizer_config = IPUFp16OptimizerHook(loss_scale=2.0, distributed=False)

    dummy_runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        timer_config=None)

    dummy_runner.call_hook('after_train_iter')
