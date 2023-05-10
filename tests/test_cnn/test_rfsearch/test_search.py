# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the rfsearch with runners.

CommandLine:
    pytest tests/test_runner/test_hooks.py
    xdoctest tests/test_hooks.py zero
"""
import logging
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.cnn.rfsearch import Conv2dRFSearchOp, RFSearchHook
from mmcv.runner import build_runner


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimizers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimizers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimizers=False):
    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimizers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


def test_rfsearchhook():

    def conv(in_channels, out_channels, kernel_size, stride, padding,
             dilation):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.stem = conv(1, 2, 3, 1, 1, 1)
            self.conv0 = conv(2, 2, 3, 1, 1, 1)
            self.layer0 = nn.Sequential(
                conv(2, 2, 3, 1, 1, 1), conv(2, 2, 3, 1, 1, 1))
            self.conv1 = conv(2, 2, 1, 1, 0, 1)
            self.conv2 = conv(2, 2, 3, 1, 1, 1)
            self.conv3 = conv(2, 2, (1, 3), 1, (0, 1), 1)

        def forward(self, x):
            x1 = self.stem(x)
            x2 = self.layer0(x1)
            x3 = self.conv0(x2)
            x4 = self.conv1(x3)
            x5 = self.conv2(x4)
            x6 = self.conv3(x5)
            return x6

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).mean(), num_samples=x.shape[0])

    rfsearch_cfg = dict(
        mode='search',
        rfstructure_file=None,
        config=dict(
            search=dict(
                step=0,
                max_step=12,
                search_interval=1,
                exp_rate=0.5,
                init_alphas=0.01,
                mmin=1,
                mmax=24,
                num_branches=2,
                skip_layer=['stem', 'conv0', 'layer0.1'])),
    )

    # hook for search
    rfsearchhook_search = RFSearchHook(
        'search', rfsearch_cfg['config'], by_epoch=True, verbose=True)
    rfsearchhook_search.config['structure'] = {
        'module.layer0.0': [1, 1],
        'module.conv2': [2, 2],
        'module.conv3': [1, 1]
    }
    # hook for fixed_single_branch
    rfsearchhook_fixed_single_branch = RFSearchHook(
        'fixed_single_branch',
        rfsearch_cfg['config'],
        by_epoch=True,
        verbose=True)
    rfsearchhook_fixed_single_branch.config['structure'] = {
        'module.layer0.0': [1, 1],
        'module.conv2': [2, 2],
        'module.conv3': [1, 1]
    }
    # hook for fixed_multi_branch
    rfsearchhook_fixed_multi_branch = RFSearchHook(
        'fixed_multi_branch',
        rfsearch_cfg['config'],
        by_epoch=True,
        verbose=True)
    rfsearchhook_fixed_multi_branch.config['structure'] = {
        'module.layer0.0': [1, 1],
        'module.conv2': [2, 2],
        'module.conv3': [1, 1]
    }

    def test_skip_layer():
        assert not isinstance(model.stem, Conv2dRFSearchOp)
        assert not isinstance(model.conv0, Conv2dRFSearchOp)
        assert isinstance(model.layer0[0], Conv2dRFSearchOp)
        assert not isinstance(model.layer0[1], Conv2dRFSearchOp)

    # 1. test init_model() with mode of search
    model = Model()
    rfsearchhook_search.init_model(model)

    test_skip_layer()
    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert isinstance(model.conv2, Conv2dRFSearchOp)
    assert isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv2.dilation_rates == [(1, 1), (3, 3)]
    assert model.conv3.dilation_rates == [(1, 1), (1, 2)]

    # 1. test step() with mode of search
    loader = DataLoader(torch.ones((1, 1, 1, 1)))
    runner = _build_demo_runner()
    runner.model = model
    runner.register_hook(rfsearchhook_search)
    runner.run([loader], [('train', 1)])

    test_skip_layer()
    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert isinstance(model.conv2, Conv2dRFSearchOp)
    assert isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv2.dilation_rates == [(1, 1), (3, 3)]
    assert model.conv3.dilation_rates == [(1, 1), (1, 3)]

    # 2. test init_model() with mode of fixed_single_branch
    model = Model()
    rfsearchhook_fixed_single_branch.init_model(model)

    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert not isinstance(model.conv2, Conv2dRFSearchOp)
    assert not isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv1.dilation == (1, 1)
    assert model.conv2.dilation == (2, 2)
    assert model.conv3.dilation == (1, 1)

    # 2. test step() with mode of fixed_single_branch
    runner = _build_demo_runner()
    runner.model = model
    runner.register_hook(rfsearchhook_fixed_single_branch)
    runner.run([loader], [('train', 1)])

    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert not isinstance(model.conv2, Conv2dRFSearchOp)
    assert not isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv1.dilation == (1, 1)
    assert model.conv2.dilation == (2, 2)
    assert model.conv3.dilation == (1, 1)

    # 3. test init_model() with mode of fixed_multi_branch
    model = Model()
    rfsearchhook_fixed_multi_branch.init_model(model)

    test_skip_layer()
    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert isinstance(model.conv2, Conv2dRFSearchOp)
    assert isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv2.dilation_rates == [(1, 1), (3, 3)]
    assert model.conv3.dilation_rates == [(1, 1), (1, 2)]

    # 3. test step() with mode of fixed_single_branch
    runner = _build_demo_runner()
    runner.model = model
    runner.register_hook(rfsearchhook_fixed_multi_branch)
    runner.run([loader], [('train', 1)])

    test_skip_layer()
    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert isinstance(model.conv2, Conv2dRFSearchOp)
    assert isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv2.dilation_rates == [(1, 1), (3, 3)]
    assert model.conv3.dilation_rates == [(1, 1), (1, 2)]
