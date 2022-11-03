# Copyright (c) OpenMMLab. All rights reserved.
import logging

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmcv.runner.fp16_utils import auto_fp16
from mmcv.utils import IS_IPU_AVAILABLE

if IS_IPU_AVAILABLE:
    from mmcv.device.ipu import cfg2options, ipu_model_wrapper
    from mmcv.device.ipu.utils import compare_ndarray

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU_AVAILABLE, reason='test case under ipu environment')


class MyBN(nn.BatchNorm2d):

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        return result, self.running_mean


# TODO Once the model training and inference interfaces
# of MMCLS and MMDET are unified,
# construct the model according to the unified standards
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = MyBN(3)
        self.relu = nn.ReLU6()
        self.fp16_enabled = False

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, return_loss=True, **kwargs):
        x = self.conv(img)
        x, running_mean = self.bn(x)
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
def test_build_model():
    for execution_strategy in \
            ['SameAsIpu', 'ShardedExecution', 'error_strategy']:
        if execution_strategy == 'error_strategy':

            def maybe_catch_error(_error):
                return pytest.raises(_error)
        else:

            class NullContextManager:

                def __enter__(self, ):
                    pass

                def __exit__(self, exc_type, exc_value, exc_traceback):
                    pass

            def maybe_catch_error(_error):
                return NullContextManager()

        with maybe_catch_error(NotImplementedError):
            options_cfg = dict(
                randomSeed=888,
                enableExecutableCaching='cache_engine',
                train_cfg=dict(
                    executionStrategy=execution_strategy,
                    Training=dict(gradientAccumulation=8),
                    availableMemoryProportion=[0.3, 0.3, 0.3, 0.3]),
                eval_cfg=dict(deviceIterations=1, ),
                partialsType='half')

            ipu_options = cfg2options(options_cfg)
            model = ToyModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            logger = logging.getLogger()
            modules_to_record = None
            ipu_model_cfg = dict(
                train_split_edges=[dict(layer_to_call='conv', ipu_id=0)],
                train_ckpt_nodes=['bn', 'conv'])
            fp16_cfg = {'loss_scale': 0.5}
            ipu_model = ipu_model_wrapper(
                model,
                ipu_options,
                optimizer,
                logger,
                modules_to_record=modules_to_record,
                ipu_model_cfg=ipu_model_cfg,
                fp16_cfg=fp16_cfg)

            ipu_model.train()
            ipu_model.eval()
            ipu_model.train()


def run_model(ipu_options,
              fp16_cfg,
              modules_to_record,
              ipu_model_wrapper_func,
              only_eval=False):
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\
        if not only_eval else None
    logger = logging.getLogger()
    ipu_model_cfg = dict(
        train_split_edges=[dict(layer_to_call='conv', ipu_id=0)],
        train_ckpt_nodes=['bn', 'conv'])
    ipu_model = ipu_model_wrapper_func(
        model,
        ipu_options,
        optimizer,
        logger,
        modules_to_record=modules_to_record,
        ipu_model_cfg=ipu_model_cfg,
        fp16_cfg=fp16_cfg)

    def get_dummy_input(training):
        if training:
            return {
                'data': {
                    'img': torch.rand((16, 3, 10, 10)),
                    'gt_label': torch.rand((16, 3, 10, 10))
                }
            }
        else:
            return {
                'img': torch.rand((16, 3, 10, 10)),
                'img_metas': {
                    'img': torch.rand((16, 3, 10, 10))
                },
                'return_loss': False
            }

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


@skip_no_ipu
def test_run_model():
    # test feature alignment not support gradientAccumulation mode
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            Training=dict(gradientAccumulation=8),
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
        partialsType='half')
    ipu_options = cfg2options(options_cfg)
    modules_to_record = ['bn']
    with pytest.raises(AssertionError, match='Feature alignment'):
        run_model(ipu_options, None, modules_to_record, ipu_model_wrapper)

    # test feature alignment not support multi-replica mode
    options_cfg = dict(
        randomSeed=888,
        replicationFactor=2,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
        partialsType='half')
    ipu_options = cfg2options(options_cfg)
    modules_to_record = ['bn']
    with pytest.raises(AssertionError, match='Feature alignment'):
        run_model(ipu_options, None, modules_to_record, ipu_model_wrapper)

    # test feature alignment not support fp16 mode
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
        partialsType='half')
    ipu_options = cfg2options(options_cfg)
    fp16_cfg = {
        'loss_scale': 0.5,
        'velocity_accum_type': 'half',
        'accum_type': 'half'
    }
    modules_to_record = ['bn']
    with pytest.raises(NotImplementedError):
        run_model(ipu_options, fp16_cfg, modules_to_record, ipu_model_wrapper)

    # test velocity_accum_type and accum_type
    fp16_cfg = {
        'loss_scale': 0.5,
        'velocity_accum_type': 'float',
        'accum_type': 'float'
    }
    run_model(ipu_options, fp16_cfg, None, ipu_model_wrapper)

    # test compile and run
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
        partialsType='half')
    ipu_options = cfg2options(options_cfg)
    modules_to_record = ['bn']
    run_model(ipu_options, None, modules_to_record, ipu_model_wrapper)

    # test feature alignment
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ))
    ipu_options = cfg2options(options_cfg)
    modules_to_record = None
    run_model(ipu_options, None, modules_to_record, ipu_model_wrapper)

    # test inference mode
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
        partialsType='half')
    ipu_options = cfg2options(options_cfg)
    fp16_cfg = {'loss_scale': 0.5}
    modules_to_record = None
    _, ipu_model = run_model(
        ipu_options,
        fp16_cfg,
        modules_to_record,
        ipu_model_wrapper,
        only_eval=True)
    with pytest.raises(RuntimeError):
        ipu_model.train()
    with pytest.raises(ValueError):
        ipu_model.train(123)
    _, ipu_model = run_model(ipu_options, None, modules_to_record,
                             ipu_model_wrapper)

    # test NotImplementedError in __call__
    ipu_model.train()
    with pytest.raises(NotImplementedError):
        ipu_model()

    # test parse_losses
    with pytest.raises(TypeError):
        ipu_model._model.model._parse_losses({'loss': None})


@skip_no_ipu
def test_compare_tensor():
    compare_ndarray(np.random.rand(3, 4), np.random.rand(3, 4))
