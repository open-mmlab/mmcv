# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

import mmcv
from mmcv.cnn import MODELS, build_model_from_cfg


def test_build_model_from_cfg():
    BACKBONES = mmcv.Registry('backbone', build_func=build_model_from_cfg)

    @BACKBONES.register_module()
    class ResNet(nn.Module):

        def __init__(self, depth, stages=4):
            super().__init__()
            self.depth = depth
            self.stages = stages

        def forward(self, x):
            return x

    @BACKBONES.register_module()
    class ResNeXt(nn.Module):

        def __init__(self, depth, stages=4):
            super().__init__()
            self.depth = depth
            self.stages = stages

        def forward(self, x):
            return x

    cfg = dict(type='ResNet', depth=50)
    model = BACKBONES.build(cfg)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = BACKBONES.build(cfg)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = [
        dict(type='ResNet', depth=50),
        dict(type='ResNeXt', depth=50, stages=3)
    ]
    model = BACKBONES.build(cfg)
    assert isinstance(model, nn.Sequential)
    assert isinstance(model[0], ResNet)
    assert model[0].depth == 50 and model[0].stages == 4
    assert isinstance(model[1], ResNeXt)
    assert model[1].depth == 50 and model[1].stages == 3

    # test inherit `build_func` from parent
    NEW_MODELS = mmcv.Registry('models', parent=MODELS, scope='new')
    assert NEW_MODELS.build_func is build_model_from_cfg

    # test specify `build_func`
    def pseudo_build(cfg):
        return cfg

    NEW_MODELS = mmcv.Registry(
        'models', parent=MODELS, build_func=pseudo_build)
    assert NEW_MODELS.build_func is pseudo_build
