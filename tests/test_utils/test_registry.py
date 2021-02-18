import pytest

import mmcv


def test_registry():
    CATS = mmcv.Registry('cat')
    assert CATS.name == 'cat'
    assert CATS.module_dict == {}
    assert len(CATS) == 0

    @CATS.register_module()
    class BritishShorthair:
        pass

    assert len(CATS) == 1
    assert CATS.get('BritishShorthair') is BritishShorthair

    class Munchkin:
        pass

    CATS.register_module(Munchkin)
    assert len(CATS) == 2
    assert CATS.get('Munchkin') is Munchkin
    assert 'Munchkin' in CATS

    with pytest.raises(KeyError):
        CATS.register_module(Munchkin)

    CATS.register_module(Munchkin, force=True)
    assert len(CATS) == 2

    # force=False
    with pytest.raises(KeyError):

        @CATS.register_module()
        class BritishShorthair:
            pass

    @CATS.register_module(force=True)
    class BritishShorthair:
        pass

    assert len(CATS) == 2

    assert CATS.get('PersianCat') is None
    assert 'PersianCat' not in CATS

    @CATS.register_module(name='Siamese')
    class SiameseCat:
        pass

    assert CATS.get('Siamese').__name__ == 'SiameseCat'

    class SphynxCat:
        pass

    CATS.register_module(name='Sphynx', module=SphynxCat)
    assert CATS.get('Sphynx') is SphynxCat

    CATS.register_module(name=['Sphynx1', 'Sphynx2'], module=SphynxCat)
    assert CATS.get('Sphynx2') is SphynxCat

    repr_str = 'Registry(name=cat, items={'
    repr_str += ("'BritishShorthair': <class 'test_registry.test_registry."
                 "<locals>.BritishShorthair'>, ")
    repr_str += ("'Munchkin': <class 'test_registry.test_registry."
                 "<locals>.Munchkin'>, ")
    repr_str += ("'Siamese': <class 'test_registry.test_registry."
                 "<locals>.SiameseCat'>, ")
    repr_str += ("'Sphynx': <class 'test_registry.test_registry."
                 "<locals>.SphynxCat'>, ")
    repr_str += ("'Sphynx1': <class 'test_registry.test_registry."
                 "<locals>.SphynxCat'>, ")
    repr_str += ("'Sphynx2': <class 'test_registry.test_registry."
                 "<locals>.SphynxCat'>")
    repr_str += '})'
    assert repr(CATS) == repr_str

    # name type
    with pytest.raises(AssertionError):
        CATS.register_module(name=7474741, module=SphynxCat)

    # the registered module should be a class
    with pytest.raises(TypeError):
        CATS.register_module(0)

    # can only decorate a class
    with pytest.raises(TypeError):

        @CATS.register_module()
        def some_method():
            pass

    # begin: test old APIs
    with pytest.warns(UserWarning):
        CATS.register_module(SphynxCat)
        assert CATS.get('SphynxCat').__name__ == 'SphynxCat'

    with pytest.warns(UserWarning):
        CATS.register_module(SphynxCat, force=True)
        assert CATS.get('SphynxCat').__name__ == 'SphynxCat'

    with pytest.warns(UserWarning):

        @CATS.register_module
        class NewCat:
            pass

        assert CATS.get('NewCat').__name__ == 'NewCat'

    with pytest.warns(UserWarning):
        CATS.deprecated_register_module(SphynxCat, force=True)
        assert CATS.get('SphynxCat').__name__ == 'SphynxCat'

    with pytest.warns(UserWarning):

        @CATS.deprecated_register_module
        class CuteCat:
            pass

        assert CATS.get('CuteCat').__name__ == 'CuteCat'

    with pytest.warns(UserWarning):

        @CATS.deprecated_register_module(force=True)
        class NewCat2:
            pass

        assert CATS.get('NewCat2').__name__ == 'NewCat2'

    # end: test old APIs


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

    # type defined using default_args
    cfg = dict(depth=50)
    model = mmcv.build_from_cfg(
        cfg, BACKBONES, default_args=dict(type='ResNet'))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(depth=50)
    model = mmcv.build_from_cfg(cfg, BACKBONES, default_args=dict(type=ResNet))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # not a registry
    with pytest.raises(TypeError):
        cfg = dict(type='VGG')
        model = mmcv.build_from_cfg(cfg, 'BACKBONES')

    # non-registered class
    with pytest.raises(KeyError):
        cfg = dict(type='VGG')
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # default_args must be a dict or None
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.build_from_cfg(cfg, BACKBONES, default_args=1)

    # cfg['type'] should be a str or class
    with pytest.raises(TypeError):
        cfg = dict(type=1000)
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # cfg should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50, stages=4)
        model = mmcv.build_from_cfg(cfg, BACKBONES)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50)
        model = mmcv.build_from_cfg(
            cfg, BACKBONES, default_args=dict(stages=4))

    # incorrect registry type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.build_from_cfg(cfg, 'BACKBONES')

    # incorrect default_args type
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = mmcv.build_from_cfg(cfg, BACKBONES, default_args=0)

    # incorrect arguments
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', non_existing_arg=50)
        model = mmcv.build_from_cfg(cfg, BACKBONES)
