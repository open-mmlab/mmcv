# Copyright (c) OpenMMLab. All rights reserved.
import logging
import numpy as np
import os.path as osp
import random
import string
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

from mmcv.runner.ipu import parse_ipu_options, build_from_cfg_with_wrapper,\
    IPU_MODE, ipu_model_wrapper, wrap_optimizer_hook,\
    IpuFp16OptimizerHook
from mmcv.runner.ipu.fp16_utils import auto_fp16
from mmcv.runner.ipu.model_converter import compare_feat


# TODO Once the model training and inference interfaces
# of MMCLS and MMDET are unified,
# construct the model according to the unified standards
class TestModel(nn.Module):
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
            return {'loss': loss}
        return x
    
    def _parse_losses(self, losses):
        return losses['loss'], losses['loss']

    def train_step(self, data, optimizer=None, **kwargs):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
        return outputs

def test_build_model():
    for executionStrategy in ['SameAsIpu', 'ShardedExecution', 'error_strategy']:
        if executionStrategy == 'error_strategy':
            maybe_catch_error = lambda _error: pytest.raises(_error)
        else:
            class NullContextManager:
                def __enter__(self,):
                    pass
                def __exit__(self, exc_type, exc_value, exc_traceback):
                    pass
            maybe_catch_error = lambda _error: NullContextManager()
        with maybe_catch_error(NotImplementedError):
            ipu_options = dict(
                randomSeed=888,
                enableExecutableCaching='cache_engine',
                train_cfgs=dict(executionStrategy=executionStrategy,
                                Training=dict(gradientAccumulation=8),
                                availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
                eval_cfgs=dict(deviceIterations=1,),
                partialsType='half')

            ipu_options = parse_ipu_options(ipu_options)
            model = TestModel()
            optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
            logger = logging.getLogger()
            modules_to_record = []
            pipeline_cfg = dict(
                train_split_edges=[
                    dict(
                        layer_to_call='conv',
                        ipu_id=0),
                ])
            fp16_cfg = {'loss_scale': 0.5}
            ipu_model = ipu_model_wrapper(
                        model, ipu_options, optimizer, logger,
                        modules_to_record=modules_to_record, pipeline_cfg=pipeline_cfg,
                        fp16_cfg=fp16_cfg)

            ipu_model.train()
            ipu_model.eval()
            ipu_model.train()


def run_model(ipu_options, fp16_cfg, modules_to_record, only_eval=False):
    model = TestModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) if not only_eval else None
    logger = logging.getLogger()
    pipeline_cfg = dict(
        train_split_edges=[
            dict(
                layer_to_call='conv',
                ipu_id=0),
        ])
    ipu_model = ipu_model_wrapper(
                model, ipu_options, optimizer, logger,
                modules_to_record=modules_to_record, pipeline_cfg=pipeline_cfg,
                fp16_cfg=fp16_cfg)
    
    def get_dummy_input(training):
        if training:
            return  {'data':{'img': torch.rand((16, 3, 10, 10)), 'gt_label': torch.rand((16, 3, 10, 10))}}
        else:
            return {'img': torch.rand((16, 3, 10, 10)), 'img_metas': {'img':torch.rand((16, 3, 10, 10))}, 'return_loss': False}
    if not only_eval:
        training = True
        ipu_model.train()
        for _ in range(3):
            dummy_input = get_dummy_input(training)
            output = ipu_model.train_step(**dummy_input)
    training = False
    ipu_model.eval()
    for _ in range(3):
        dummy_input = get_dummy_input(training)
        output = ipu_model(**dummy_input)
    return output, ipu_model


def test_run_model():

    # test feature alignment not support gradientAccumulation mode
    ipu_options = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        Training=dict(gradientAccumulation=8),
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    modules_to_record = ['bn']
    with pytest.raises(
            AssertionError,
            match='Feature alignment for gradient accumulation mode is not implemented'):
        run_model(ipu_options, None, modules_to_record)

    # test feature alignment not support multi-replica mode
    ipu_options = dict(
        randomSeed=888,
        replicationFactor=2,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    modules_to_record = ['bn']
    with pytest.raises(
            AssertionError,
            match='Feature alignment for multi-replica mode is not implemented'):
        run_model(ipu_options, None, modules_to_record)

    # test feature alignment not support fp16 mode
    ipu_options = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    fp16_cfg = {'loss_scale': 0.5}
    modules_to_record = ['bn']
    with pytest.raises(NotImplementedError):
        run_model(ipu_options, fp16_cfg, modules_to_record)

    # test compile and run
    ipu_options = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    modules_to_record = ['bn']
    run_model(ipu_options, None, modules_to_record)

    # test feature alignment
    ipu_options = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    fp16_cfg = {'loss_scale': 0.5}
    modules_to_record = []
    run_model(ipu_options, fp16_cfg, modules_to_record)

    # test inference mode
    ipu_options = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),
        partialsType='half')
    ipu_options = parse_ipu_options(ipu_options)
    fp16_cfg = {'loss_scale': 0.5}
    modules_to_record = []
    _, ipu_model = run_model(ipu_options, fp16_cfg, modules_to_record, only_eval=True)
    with pytest.raises(RuntimeError):
        ipu_model.train()
    with pytest.raises(ValueError):
        ipu_model.train(123)
    _, ipu_model = run_model(ipu_options, fp16_cfg, modules_to_record)

    # check NotImplementedError in __call__
    ipu_model.train()
    with pytest.raises(NotImplementedError):
        ipu_model()


def test_compare_feat():
    compare_feat(np.random.rand(3, 4), np.random.rand(3, 4))
