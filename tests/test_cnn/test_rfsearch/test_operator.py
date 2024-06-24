# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn

from mmcv.cnn.rfsearch.operator import Conv2dRFSearchOp

global_config = dict(
    step=0,
    max_step=12,
    search_interval=1,
    exp_rate=0.5,
    init_alphas=0.01,
    mmin=1,
    mmax=24,
    num_branches=2,
    skip_layer=['stem', 'layer1'])


# test with 3x3 conv
def test_rfsearch_operator_3x3():
    conv = nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    operator = Conv2dRFSearchOp(conv, global_config)
    x = torch.randn(1, 3, 32, 32)

    # set no_grad to perform in-place operator
    with torch.no_grad():
        # After expand: (1, 1) (2, 2)
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (2, 2)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After estimate: (2, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (2, 2)
        assert operator.op_layer.dilation == (2, 2)
        assert operator.op_layer.padding == (2, 2)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (3, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.1
        operator.branch_weights[1] = 0.4
        # After estimate: (3, 3) with branch_weights of [0.2 0.8]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (3, 3)
        assert operator.op_layer.dilation == (3, 3)
        assert operator.op_layer.padding == (3, 3)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)


# test with 5x5 conv
def test_rfsearch_operator_5x5():
    conv = nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2)
    operator = Conv2dRFSearchOp(conv, global_config)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        # After expand: (1, 1) (2, 2)
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (2, 2)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After estimate: (2, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (2, 2)
        assert operator.op_layer.dilation == (2, 2)
        assert operator.op_layer.padding == (4, 4)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (3, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.1
        operator.branch_weights[1] = 0.4
        # After estimate: (3, 3) with branch_weights of [0.2 0.8]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (3, 3)
        assert operator.op_layer.dilation == (3, 3)
        assert operator.op_layer.padding == (6, 6)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)


# test with 5x5 conv num_branches=3
def test_rfsearch_operator_5x5_branch3():
    conv = nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2)
    config = deepcopy(global_config)
    config['num_branches'] = 3
    operator = Conv2dRFSearchOp(conv, config)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        # After expand: (1, 1) (2, 2)
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (2, 2)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After estimate: (2, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (2, 2)
        assert operator.op_layer.dilation == (2, 2)
        assert operator.op_layer.padding == (4, 4)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (2, 2) (3, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 3
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (2, 2)
        assert operator.dilation_rates[2] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.1
        operator.branch_weights[1] = 0.3
        operator.branch_weights[2] = 0.6
        # After estimate: (3, 3) with branch_weights of [0.1 0.3 0.6]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (3, 3)
        assert operator.op_layer.dilation == (3, 3)
        assert operator.op_layer.padding == (6, 6)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)


# test with 1x5 conv
def test_rfsearch_operator_1x5():
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=(1, 5),
        stride=1,
        padding=(0, 2))
    operator = Conv2dRFSearchOp(conv, global_config)
    x = torch.randn(1, 3, 32, 32)

    # After expand: (1, 1) (1, 2)
    assert len(operator.dilation_rates) == 2
    assert operator.dilation_rates[0] == (1, 1)
    assert operator.dilation_rates[1] == (1, 2)
    assert torch.all(
        operator.branch_weights.data == global_config['init_alphas']).item()
    # test forward
    assert operator(x).shape == (1, 3, 32, 32)

    with torch.no_grad():
        # After estimate: (1, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (1, 2)
        assert operator.op_layer.dilation == (1, 2)
        assert operator.op_layer.padding == (0, 4)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (1, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (1, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.2
        operator.branch_weights[1] = 0.8
        # After estimate: (3, 3) with branch_weights of [0.2 0.8]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (1, 3)
        assert operator.op_layer.dilation == (1, 3)
        assert operator.op_layer.padding == (0, 6)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)


# test with 5x5 conv initial_dilation=(2, 2)
def test_rfsearch_operator_5x5_d2x2():
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=5,
        stride=1,
        padding=4,
        dilation=(2, 2))
    operator = Conv2dRFSearchOp(conv, global_config)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        # After expand: (1, 1) (3, 3)
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After estimate: (2, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (2, 2)
        assert operator.op_layer.dilation == (2, 2)
        assert operator.op_layer.padding == (4, 4)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (3, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.8
        operator.branch_weights[1] = 0.2
        # After estimate: (3, 3) with branch_weights of [0.8 0.2]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.op_layer.dilation == (1, 1)
        assert operator.op_layer.padding == (2, 2)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)


# test with 5x5 conv initial_dilation=(1, 2)
def test_rfsearch_operator_5x5_d1x2():
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel_size=5,
        stride=1,
        padding=(2, 4),
        dilation=(1, 2))
    operator = Conv2dRFSearchOp(conv, global_config)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        # After expand: (1, 1) (2, 3)
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (2, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After estimate: (2, 2) with branch_weights of [0.5 0.5]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (2, 2)
        assert operator.op_layer.dilation == (2, 2)
        assert operator.op_layer.padding == (4, 4)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        # After expand: (1, 1) (3, 3)
        operator.expand_rates()
        assert len(operator.dilation_rates) == 2
        assert operator.dilation_rates[0] == (1, 1)
        assert operator.dilation_rates[1] == (3, 3)
        assert torch.all(operator.branch_weights.data ==
                         global_config['init_alphas']).item()
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)

        operator.branch_weights[0] = 0.1
        operator.branch_weights[1] = 0.8
        # After estimate: (3, 3) with branch_weights of [0.1 0.8]
        operator.estimate_rates()
        assert len(operator.dilation_rates) == 1
        assert operator.dilation_rates[0] == (3, 3)
        assert operator.op_layer.dilation == (3, 3)
        assert operator.op_layer.padding == (6, 6)
        # test forward
        assert operator(x).shape == (1, 3, 32, 32)
