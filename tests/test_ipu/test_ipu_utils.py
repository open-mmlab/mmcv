# Copyright (c) OpenMMLab. All rights reserved.
import copy
import pytest

import mmcv
from mmcv.utils.ipu_wrapper import IPU_MODE
if IPU_MODE:
    from mmcv.runner.ipu.util import parse_ipu_options

skip_no_ipu = pytest.mark.skipif(
    not IPU_MODE, reason='test case under ipu environment')


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
    model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNet', depth=50)
    model = mmcv.runner.ipu.\
        build_from_cfg_with_wrapper(cfg, BACKBONES, default_args={'stages': 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type=ResNet, depth=50)
    model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # type defined using default_args
    cfg = dict(depth=50)
    model = mmcv.runner.ipu.build_from_cfg_with_wrapper(
        cfg, BACKBONES, default_args=dict(type='ResNet'))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(depth=50)
    model = mmcv.runner.ipu.\
        build_from_cfg_with_wrapper(cfg,
                                    BACKBONES,
                                    default_args=dict(type=ResNet))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # not a registry
    with pytest.raises(TypeError):
        cfg = dict(type='VGG')
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, 'BACKBONES')

    # non-registered class
    with pytest.raises(KeyError):
        cfg = dict(type='VGG')
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)

    # default_args must be a dict or None
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.runner.ipu.\
            build_from_cfg_with_wrapper(cfg, BACKBONES, default_args=1)

    # cfg['type'] should be a str or class
    with pytest.raises(TypeError):
        cfg = dict(type=1000)
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)

    # cfg should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50, stages=4)
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50)
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(
            cfg, BACKBONES, default_args=dict(stages=4))

    # incorrect registry type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, 'BACKBONES')

    # incorrect default_args type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.runner.\
            ipu.build_from_cfg_with_wrapper(cfg, BACKBONES, default_args=0)

    # incorrect arguments
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', non_existing_arg=50)
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)

    # cfg not dict
    with pytest.raises(TypeError):
        cfg = []
        model = mmcv.runner.ipu.build_from_cfg_with_wrapper(cfg, BACKBONES)


@skip_no_ipu
def test_parse_ipu_options():
    options_cfg = dict(
        randomSeed=888,
        enableExecutableCaching='cache_engine',
        train_cfgs=dict(executionStrategy='SameAsIpu',
                        Training=dict(gradientAccumulation=8),
                        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],),
        eval_cfgs=dict(deviceIterations=1,),)
    parse_ipu_options(copy.deepcopy(options_cfg))

    with pytest.raises(
            NotImplementedError,
            match='cfg type'):
        _options_cfg = copy.deepcopy(options_cfg)
        _options_cfg['randomSeed'] = (1, 3)
        parse_ipu_options(_options_cfg)

    with pytest.raises(
            NotImplementedError,
            match='opts_node type'):
        _options_cfg = copy.deepcopy(options_cfg)
        _options_cfg['train_cfgs']['Precision'] = {'autocast_policy': 123}
        parse_ipu_options(_options_cfg)
