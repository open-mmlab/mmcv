# Copyright (c) OpenMMLab. All rights reserved.
"""Tests the rfsearch with runners.

CommandLine:
    pytest tests/test_runner/test_hooks.py
    xdoctest tests/test_hooks.py zero
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mmcv.cnn.rfsearch import Conv2dRFSearchOp, RFSearchHook
from tests.test_runner.test_hooks import _build_demo_runner


def test_rfsearchhook():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1)
            self.conv2 = nn.Conv2d(
                in_channels=2,
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1)
            self.conv3 = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                dilation=1)

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

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
                skip_layer=['stem', 'layer1'])),
    )

    # hook for search
    rfsearchhook_search = RFSearchHook(
        'search', rfsearch_cfg['config'], by_epoch=True, verbose=True)
    rfsearchhook_search.config['structure'] = {
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
        'module.conv2': [2, 2],
        'module.conv3': [1, 1]
    }

    # 1. test init_model() with mode of search
    model = Model()
    rfsearchhook_search.init_model(model)

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

    assert not isinstance(model.conv1, Conv2dRFSearchOp)
    assert isinstance(model.conv2, Conv2dRFSearchOp)
    assert isinstance(model.conv3, Conv2dRFSearchOp)
    assert model.conv2.dilation_rates == [(1, 1), (3, 3)]
    assert model.conv3.dilation_rates == [(1, 1), (1, 2)]
