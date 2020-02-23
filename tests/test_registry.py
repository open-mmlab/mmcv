import pytest

import mmcv


def test_registry():
    reg_name = 'cat'
    CATS = mmcv.Registry(reg_name)
    assert CATS.name == reg_name
    assert CATS.module_dict == {}
    assert len(CATS) == 0

    @CATS.register_module
    class BritishShorthair:
        pass

    assert len(CATS) == 1
    assert CATS.get('BritishShorthair') is BritishShorthair

    class Munchkin:
        pass

    CATS.register_module(Munchkin)
    assert len(CATS) == 2
    assert CATS.get('Munchkin') is Munchkin

    with pytest.raises(KeyError):
        CATS.register_module(Munchkin)

    CATS.register_module(Munchkin)
    assert len(CATS) == 2

    @CATS.register_module(force=True)
    class BritishShorthair:
        pass

    assert len(CATS) == 2

    assert CATS.get('PersianCat') is None

    assert repr(
        CATS) == "Registry(name=cat, items=['BritishShorthair', 'Munchkin'])"


def test_build_from_cfg():
    BACKBONES = mmcv.Registry('backbone')

    @BACKBONES.register_module
    class ResNet:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    @BACKBONES.register_module
    class ResNeXt:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    cfg = dict(type='ResNet', depth=50)
    model = mmcv.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNet', depth=50)
    model = mmcv.build_from_cfg(cfg, BACKBONES, default_args={'stages': 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = mmcv.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type=ResNet, depth=50)
    model = mmcv.build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # non-registered class
    with pytest.raises(KeyError):
        cfg = dict(type='VGG')
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # cfg['type'] should be a str or class
    with pytest.raises(TypeError):
        cfg = dict(type=1000)
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # cfg should contain the key "type"
    with pytest.raises(TypeError):
        cfg = dict(depth=50, stages=4)
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # incorrect registry type
    with pytest.raises(TypeError):
        dict(type='ResNet', depth=50)
        model = mmcv.build_from_cfg(cfg, 'BACKBONES')

    # incorrect default_args type
    with pytest.raises(TypeError):
        dict(type='ResNet', depth=50)
        model = mmcv.build_from_cfg(cfg, BACKBONES, default_args=0)
