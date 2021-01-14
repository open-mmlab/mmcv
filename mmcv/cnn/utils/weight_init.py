# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import torch.nn as nn

from mmcv.utils import (build_from_cfg, get_logger, print_log, Registry)

INITIALIZERS = Registry('initializer')


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        bias=bias,
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


class BaseInit(object):

    def __init__(self, bias, layers):
        if not isinstance(bias, (dict, int, float)):
            raise TypeError(
                f'bias must be a dict or numbel, but got {type(bias)}')
        if isinstance(bias, dict):
            func = build_from_cfg(bias, INITIALIZERS)
            self.bias = func()
        else:
            self.bias = bias
        self.layers = layers


@INITIALIZERS.register_module(name='Constant')
class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.

    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float | dict, optional): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        layers (str |  [str], optional): the layer will be initialized.
        Defaults to None.
    """

    def __init__(self, val, bias=0, layers=None):
        super().__init__(bias, layers)
        self.val = val

    def __call__(self, module):
        if self.layers is None:
            constant_init(module, self.val, self.bias)
        else:
            if isinstance(self.layers, str):
                layername = module.__class__.__name__
                if layername == self.layers:
                    constant_init(module, self.val, self.bias)
            elif isinstance(self.layers, list):
                for layer in self.layers:
                    layername = module.__class__.__name__
                    if layername == layer:
                        constant_init(module, self.val, self.bias)
            else:
                raise TypeError(
                    f'layers must be str or [str], but gor {type(self.layers)}'
                )


@INITIALIZERS.register_module(name='Xavier')
class XavierInit(BaseInit):
    """Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010).

    Args:
        gain (int | float, optional): an optional scaling factor. Defaults
        to 1.
        bias (int | float | dict, optional): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        distribution (str, optional): distribution either be ``'normal'``
        or ``'uniform'``. Defaults to ``'normal'``.
        layers (str | [str], optional): the layer will be initialized.
        Defaults to None.
    """

    def __init__(self, gain=1, bias=0, distribution='normal', layers=None):
        super().__init__(bias, layers)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module):
        if self.layers is None:
            xavier_init(module, self.gain, self.bias, self.distribution)
        else:
            if isinstance(self.layers, str):
                layername = module.__class__.__name__
                if layername == self.layers:
                    xavier_init(module, self.gain, self.bias,
                                self.distribution)
            elif isinstance(self.layers, list):
                for layer in self.layers:
                    layername = module.__class__.__name__
                    if layername == layer:
                        xavier_init(module, self.gain, self.bias,
                                    self.distribution)
            else:
                raise TypeError(
                    f'layers must be str or [str], but gor {type(self.layers)}'
                )


@INITIALIZERS.register_module(name='Normal')
class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float, optional):the mean of the normal distribution.
        Defaults to 0.
        std (int | float, optional): the standard deviation of the normal
        distribution. Defaults to 1.
        bias (int | float | dict, optional): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        layers (str | [str], optional): the layer will be initialized.
        Defaults to None.

    """

    def __init__(self, mean=0, std=1, bias=0, layers=None):
        super().__init__(bias, layers)
        self.mean = mean
        self.std = std

    def __call__(self, module):
        if self.layers is None:
            normal_init(module, self.mean, self.std, self.bias)
        else:
            if isinstance(self.layers, str):
                layername = module.__class__.__name__
                if layername == self.layers:
                    normal_init(module, self.mean, self.std, self.bias)
            elif isinstance(self.layers, list):
                for layer in self.layers:
                    layername = module.__class__.__name__
                    if layername == layer:
                        normal_init(module, self.mean, self.std, self.bias)
            else:
                raise TypeError(
                    f'layers must be str or [str], but gor {type(self.layers)}'
                )


@INITIALIZERS.register_module(name='Uniform')
class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        a (int | float, optional): the lower bound of the uniform
        distribution. Defaults to 0.
        b (int | float, optional): the upper bound of the uniform
        distribution. Defaults to 1.
        bias (int | float | dict, optional): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        layers (str | [str], optional): the layer will be initialized.
        Defaults to None.
    """

    def __init__(self, a=0, b=1, bias=0, layers=None):
        super().__init__(bias, layers)
        self.a = a
        self.b = b

    def __call__(self, module):
        if self.layers is None:
            uniform_init(module, self.a, self.b, self.bias)
        else:
            if isinstance(self.layers, str):
                layername = module.__class__.__name__
                if layername == self.layers:
                    uniform_init(module, self.a, self.b, self.bias)
            elif isinstance(self.layers, list):
                for layer in self.layers:
                    layername = module.__class__.__name__
                    if layername == layer:
                        uniform_init(module, self.a, self.b, self.bias)
            else:
                raise TypeError(
                    f'layers must be str or [str], but gor {type(self.layers)}'
                )


@INITIALIZERS.register_module(name='Kaiming')
class KaimingInit(BaseInit):
    """Initialize module paramters with the valuse according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015).

    Args:
        a (int | float, optional): the negative slope of the rectifier used
        after this layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str, optional):  either ``'fan_in'`` or ``'fan_out'``.
        Choosing ``'fan_in'`` preserves the magnitude of the variance of
        the weights in the forward pass. Choosing ``'fan_out'`` preserves
        the magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str, optional): the non-linear function
        (`nn.functional` name), recommended to use only with ``'relu'`` or
        ``'leaky_relu'`` . Defaults to 'relu'.
        bias (int | float | dict, optional): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        distribution (str, optional): distribution either be ``'normal'``
        or ``'uniform'``. Defaults to ``'normal'``.
        layers (str | [str], optional): the layer will be initialized.
        Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal',
                 layers=None):
        super().__init__(bias, layers)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):
        if self.layers is None:
            kaiming_init(module, self.a, self.mode, self.nonlinearity,
                         self.bias, self.distribution)
        else:
            if isinstance(self.layers, str):
                layername = module.__class__.__name__
                if layername == self.layers:
                    kaiming_init(module, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)
            elif isinstance(self.layers, list):
                for layer in self.layers:
                    layername = module.__class__.__name__
                    if layername == layer:
                        kaiming_init(module, self.a, self.mode,
                                     self.nonlinearity, self.bias,
                                     self.distribution)
            else:
                raise TypeError(
                    f'layers must be str or [str], but gor {type(self.layers)}'
                )


@INITIALIZERS.register_module(name='BiasProb')
class BiasInitWithProb(object):
    """Initialize conv/fc bias value according to giving probablity.
    Args:
        prior_prob (float): value as prior probability
    """

    def __init__(self, prior_prob):
        self.prior_prob = prior_prob

    def __call__(self):
        return bias_init_with_prob(self.prior_prob)


@INITIALIZERS.register_module(name='Pretrained')
class PretrainedInit(object):
    """Initialize module by loading a pretrained model
    Args:
        checkpoint (str): the file should be load
        prefix (str, optional): the prefix to indicate the sub-module.
        Defaults to None.
    """

    def __init__(self, checkpoint, prefix=None, map_location=None):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def __call__(self, module):
        from mmcv.runner import (_load_checkpoint_with_prefix, load_checkpoint,
                                 load_state_dict)
        logger = get_logger('mmcv')
        if self.prefix is None:
            print_log(f'load model from: {self.checkpoint}', logger=logger)
            load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger=logger)
        else:
            print_log(
                f'load {self.prefix} in model from: {self.checkpoint}',
                logger=logger)
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)
            load_state_dict(module, state_dict, strict=False, logger=logger)


def _initialize(module, cfg):
    func = build_from_cfg(cfg, INITIALIZERS)
    if cfg.get('type') == 'PretrainedInit':
        func(module)
    else:
        module.apply(func)


def _initialize_cases(module, cases):
    if isinstance(cases, list):
        for case in cases:
            name = case.pop('name', None)
            if hasattr(module, name):
                _initialize(getattr(module, name), case)
            else:
                raise RuntimeError(f'module did not have attribute {name}')

    elif isinstance(cases, dict):
        name = cases.pop('name', None)
        if hasattr(module, name):
            _initialize(getattr(module, name), cases)
        else:
            raise RuntimeError(f'module did not have attribute {name}')
    else:
        raise TypeError(f'cases must be a dict or list, but got {type(cases)}')


def initialize(module, init_cfg):
    """Initialize a module

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization config dict to define
        initializer. OpenMMLab has implemented 7 initializers including
        ``Constant``, ``Xavier`, ``Normal``, ``Uniform``,
        ``Kaiming``, ``Pretrained`` and ``BiasProb`` for
        bias initialization.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', val =1 , bias =2)
        >>> initialize(module, init_cfg)
        >>> for p in module.parameters():
                print(p)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.]], requires_grad=True)
            Parameter containing:
            tensor([2., 2., 2.], requires_grad=True)

        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layers'`` for initializing layers with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layers='Conv1d', val=1),
                dict(type='ConstantInit', layers='Linear', val=2)]
        >>> initialize(module, init_cfg)
        >>> for name, param in module.named_parameters():
                print(name, param)
        0.weight Parameter containing:
        tensor([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]], requires_grad=True)
        0.bias Parameter containing:
        tensor([0.], requires_grad=True)
        1.weight Parameter containing:
        tensor([[2.],
                [2.]], requires_grad=True)
        1.bias Parameter containing:
        tensor([0., 0.], requires_grad=True)

        >>> # Omitting ``'layers'`` initialize module with same configuration
        >>> init_cfg = dict(type='Constant', val=1, bias=2)
        >>> initialize(module, init_cfg)
        >>> for name, param in module.named_parameters():
                print(name, param)
        0.weight Parameter containing:
        tensor([[[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]], requires_grad=True)
        0.bias Parameter containing:
        tensor([2.], requires_grad=True)
        1.weight Parameter containing:
        tensor([[1.],
                [1.]], requires_grad=True)
        1.bias Parameter containing:
        tensor([2., 2.], requires_grad=True)

        >>> # define key``'cases'`` to initialize some specific cases in module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2,
        >>>     cases=dict(type='Constant', name='reg', val=3, bias=4))
        >>> initialize(model, init_cfg)

        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='PretrainedInit',
                checkpoint='torchvision://resnet50')
        >>> initialize(model, init_cfg)

        >>> # Intialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='Pretrained',
                checkpoint=url, prefix='backbone.')
    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(f'init_cfg must be a dict, but got {type(init_cfg)}')

    if isinstance(init_cfg, dict):
        cases = init_cfg.pop('cases', None)
        _initialize(module, init_cfg)

        if cases is not None:
            _initialize_cases(module, cases)
        else:
            # All attributes in module have same initialization.
            pass

    else:
        for cfg in init_cfg:
            cases = cfg.pop('cases', None)
            _initialize(module, cfg)

            if cases is not None:
                _initialize_cases(module, cases)
            else:
                # All attributes in module have same initialization.
                pass
