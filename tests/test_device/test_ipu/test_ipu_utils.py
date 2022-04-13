# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest
import torch.nn as nn

import mmcv
from mmcv.device.ipu import IS_IPU

if IS_IPU:
    from mmcv.device.ipu import cast_to_options
    from mmcv.device.ipu.utils import model_sharding

skip_no_ipu = pytest.mark.skipif(
    not IS_IPU, reason='test case under ipu environment')


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU6()


@skip_no_ipu
def test_build_from_cfg():
    BACKBONES = mmcv.Registry('backbone')

    @BACKBONES.register_module()
    class ResNet:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    @BACKBONES.register_module()
    class ResNeXt:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    cfg = dict(type='ResNet', depth=50)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNet', depth=50)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
        cfg, BACKBONES, default_args={'stages': 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type=ResNet, depth=50)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # type defined using default_args
    cfg = dict(depth=50)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
        cfg, BACKBONES, default_args=dict(type='ResNet'))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(depth=50)
    model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
        cfg, BACKBONES, default_args=dict(type=ResNet))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # not a registry
    with pytest.raises(TypeError):
        cfg = dict(type='VGG')
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, 'BACKBONES')

    # non-registered class
    with pytest.raises(KeyError):
        cfg = dict(type='VGG')
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES)

    # default_args must be a dict or None
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES, default_args=1)

    # cfg['type'] should be a str or class
    with pytest.raises(TypeError):
        cfg = dict(type=1000)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES)

    # cfg should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50, stages=4)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES, default_args=dict(stages=4))

    # incorrect registry type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, 'BACKBONES')

    # incorrect default_args type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES, default_args=0)

    # incorrect arguments
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', non_existing_arg=50)
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES)

    # cfg not dict
    with pytest.raises(TypeError):
        cfg = []
        model = mmcv.device.ipu.runner.build_from_cfg_with_wrapper(
            cfg, BACKBONES)


@skip_no_ipu
def test_parse_ipu_options():
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfg=dict(
            executionStrategy='SameAsIpu',
            Training=dict(gradientAccumulation=8),
            availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
        ),
        eval_cfg=dict(deviceIterations=1, ),
    )
    cast_to_options(copy.deepcopy(options_cfg))

    with pytest.raises(NotImplementedError, match='cfg type'):
        _options_cfg = copy.deepcopy(options_cfg)
        _options_cfg['randomSeed'] = (1, 3)
        cast_to_options(_options_cfg)

    with pytest.raises(NotImplementedError, match='options_node type'):
        _options_cfg = copy.deepcopy(options_cfg)
        _options_cfg['train_cfg']['Precision'] = {'autocast_policy': 123}
        cast_to_options(_options_cfg)


@skip_no_ipu
def test_model_sharding():

    model = ToyModel()
    split_edges = [dict(layer_to_call='666', ipu_id=0)]

    with pytest.raises(RuntimeError, match='split_edges:'):
        model_sharding(model, split_edges)

    model = ToyModel()
    split_edges = [
        dict(layer_to_call='conv', ipu_id=0),
        dict(layer_to_call=1, ipu_id=0)
    ]

    with pytest.raises(ValueError, match='The same layer is referenced'):
        model_sharding(model, split_edges)
