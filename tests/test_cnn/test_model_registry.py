import torch.nn as nn

import mmcv
from mmcv.cnn import build_model_from_cfg


def test_build_model_from_cfg():
    BACKBONES = mmcv.Registry('backbone', build_func=build_model_from_cfg)

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
    model = BACKBONES.build(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = BACKBONES.build(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = [
        dict(type='ResNet', depth=50),
        dict(type='ResNeXt', depth=50, stages=3)
    ]
    model = BACKBONES.build(cfg, BACKBONES)
    assert isinstance(model, nn.Sequential)
    assert isinstance(model[0], ResNet)
    assert model[0].depth == 50 and model[0].stages == 4
    assert isinstance(model[1], ResNeXt)
    assert model[1].depth == 50 and model[1].stages == 3
