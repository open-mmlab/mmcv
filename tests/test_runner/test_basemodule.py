import torch
from torch import nn

from mmcv.runner import BaseModule
from mmcv.utils import Registry, build_from_cfg

COMPONENTS = Registry('component')
FOOMODELS = Registry('model')


@COMPONENTS.register_module()
class FooConv1d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1d = nn.Conv1d(4, 1, 4)

    def forward(self, x):
        return self.conv1d(x)


@COMPONENTS.register_module()
class FooConv2d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv2d = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        return self.conv2d(x)


@COMPONENTS.register_module()
class FooLinear(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)


@COMPONENTS.register_module()
class FooLinearConv1d(BaseModule):

    def __init__(self, linear=None, conv1d=None, init_cfg=None):
        super().__init__(init_cfg)
        if linear is not None:
            self.linear = build_from_cfg(linear, COMPONENTS)
        if conv1d is not None:
            self.conv1d = build_from_cfg(conv1d, COMPONENTS)

    def forward(self, x):
        x = self.linear(x)
        return self.conv1d(x)


@FOOMODELS.register_module()
class FooModel(BaseModule):

    def __init__(self,
                 component1=None,
                 component2=None,
                 component3=None,
                 component4=None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg)
        if component1 is not None:
            self.component1 = build_from_cfg(component1, COMPONENTS)
        if component2 is not None:
            self.component2 = build_from_cfg(component2, COMPONENTS)
        if component3 is not None:
            self.component3 = build_from_cfg(component3, COMPONENTS)
        if component4 is not None:
            self.component4 = build_from_cfg(component4, COMPONENTS)

        # its type is not BaseModule, it can be initialized
        # with "override" key.
        self.reg = nn.Linear(3, 4)


def test_model_weight_init():
    """
    Config
    model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                     Conv2d: weight=5, bias=6)
    ├──component1 (FooConv1d)
    ├──component2 (FooConv2d)
    ├──component3 (FooLinear)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear)
        ├──conv1d (FooConv1d)
    ├──reg (nn.Linear)

    Parameters after initialization
    model (FooModel)
    ├──component1 (FooConv1d, weight=3, bias=4)
    ├──component2 (FooConv2d, weight=5, bias=6)
    ├──component3 (FooLinear, weight=1, bias=2)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear, weight=1, bias=2)
        ├──conv1d (FooConv1d, weight=3, bias=4)
    ├──reg (nn.Linear, weight=1, bias=2)
    """
    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(type='Constant', val=1, bias=2, layer='Linear'),
            dict(type='Constant', val=3, bias=4, layer='Conv1d'),
            dict(type='Constant', val=5, bias=6, layer='Conv2d')
        ],
        component1=dict(type='FooConv1d'),
        component2=dict(type='FooConv2d'),
        component3=dict(type='FooLinear'),
        component4=dict(
            type='FooLinearConv1d',
            linear=dict(type='FooLinear'),
            conv1d=dict(type='FooConv1d')))

    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weight()

    assert torch.equal(model.component1.conv1d.weight,
                       torch.full(model.component1.conv1d.weight.shape, 3.0))
    assert torch.equal(model.component1.conv1d.bias,
                       torch.full(model.component1.conv1d.bias.shape, 4.0))
    assert torch.equal(model.component2.conv2d.weight,
                       torch.full(model.component2.conv2d.weight.shape, 5.0))
    assert torch.equal(model.component2.conv2d.bias,
                       torch.full(model.component2.conv2d.bias.shape, 6.0))
    assert torch.equal(model.component3.linear.weight,
                       torch.full(model.component3.linear.weight.shape, 1.0))
    assert torch.equal(model.component3.linear.bias,
                       torch.full(model.component3.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.linear.linear.weight,
        torch.full(model.component4.linear.linear.weight.shape, 1.0))
    assert torch.equal(
        model.component4.linear.linear.bias,
        torch.full(model.component4.linear.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.weight,
        torch.full(model.component4.conv1d.conv1d.weight.shape, 3.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.bias,
        torch.full(model.component4.conv1d.conv1d.bias.shape, 4.0))
    assert torch.equal(model.reg.weight, torch.full(model.reg.weight.shape,
                                                    1.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 2.0))


def test_nest_components_weight_init():
    """
    Config
    model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                     Conv2d: weight=5, bias=6)
    ├──component1 (FooConv1d, Conv1d: weight=7, bias=8)
    ├──component2 (FooConv2d, Conv2d: weight=9, bias=10)
    ├──component3 (FooLinear)
    ├──component4 (FooLinearConv1d, Linear: weight=11, bias=12)
        ├──linear (FooLinear, Linear: weight=11, bias=12)
        ├──conv1d (FooConv1d)
    ├──reg (nn.Linear, weight=13, bias=14)

    Parameters after initialization
    model (FooModel)
    ├──component1 (FooConv1d, weight=7, bias=8)
    ├──component2 (FooConv2d, weight=9, bias=10)
    ├──component3 (FooLinear, weight=1, bias=2)
    ├──component4 (FooLinearConv1d)
        ├──linear (FooLinear, weight=1, bias=2)
        ├──conv1d (FooConv1d, weight=3, bias=4)
    ├──reg (nn.Linear, weight=13, bias=14)
    """

    model_cfg = dict(
        type='FooModel',
        init_cfg=[
            dict(
                type='Constant',
                val=1,
                bias=2,
                layer='Linear',
                override=dict(type='Constant', name='reg', val=13, bias=14)),
            dict(type='Constant', val=3, bias=4, layer='Conv1d'),
            dict(type='Constant', val=5, bias=6, layer='Conv2d'),
        ],
        component1=dict(
            type='FooConv1d', init_cfg=dict(type='Constant', val=7, bias=8)),
        component2=dict(
            type='FooConv2d', init_cfg=dict(type='Constant', val=9, bias=10)),
        component3=dict(type='FooLinear'),
        component4=dict(
            type='FooLinearConv1d',
            linear=dict(type='FooLinear'),
            conv1d=dict(type='FooConv1d')))

    model = build_from_cfg(model_cfg, FOOMODELS)
    model.init_weight()

    assert torch.equal(model.component1.conv1d.weight,
                       torch.full(model.component1.conv1d.weight.shape, 7.0))
    assert torch.equal(model.component1.conv1d.bias,
                       torch.full(model.component1.conv1d.bias.shape, 8.0))
    assert torch.equal(model.component2.conv2d.weight,
                       torch.full(model.component2.conv2d.weight.shape, 9.0))
    assert torch.equal(model.component2.conv2d.bias,
                       torch.full(model.component2.conv2d.bias.shape, 10.0))
    assert torch.equal(model.component3.linear.weight,
                       torch.full(model.component3.linear.weight.shape, 1.0))
    assert torch.equal(model.component3.linear.bias,
                       torch.full(model.component3.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.linear.linear.weight,
        torch.full(model.component4.linear.linear.weight.shape, 1.0))
    assert torch.equal(
        model.component4.linear.linear.bias,
        torch.full(model.component4.linear.linear.bias.shape, 2.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.weight,
        torch.full(model.component4.conv1d.conv1d.weight.shape, 3.0))
    assert torch.equal(
        model.component4.conv1d.conv1d.bias,
        torch.full(model.component4.conv1d.conv1d.bias.shape, 4.0))
    assert torch.equal(model.reg.weight,
                       torch.full(model.reg.weight.shape, 13.0))
    assert torch.equal(model.reg.bias, torch.full(model.reg.bias.shape, 14.0))
