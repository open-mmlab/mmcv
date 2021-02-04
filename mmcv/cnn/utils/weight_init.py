# Copyright (c) Open-MMLab. All rights reserved.
import numpy as np
import torch.nn as nn

from mmcv.utils import Registry, build_from_cfg, get_logger, print_log

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

    def __init__(self, bias, layer):
        if not isinstance(bias, (dict, int, float)):
            raise TypeError(
                f'bias must be a dict or numbel, but got {type(bias)}')
        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be str or list[str], \
                    but got {type(layer)}')

        if isinstance(bias, dict):
            func = build_from_cfg(bias, INITIALIZERS)
            self.bias = func()
        else:
            self.bias = bias
        if isinstance(layer, str):
            self.layer = [layer]
        else:
            self.layer = layer


@INITIALIZERS.register_module(name='Constant')
class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.

    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float | dict): the value to fill the bias or
        define initialization type for bias. Defaults to 0.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, val, bias=0, layer=None):
        super().__init__(bias, layer)
        self.val = val

    def __call__(self, module):
        if self.layer is None:
            constant_init(module, self.val, self.bias)
        else:
            for layer_ in self.layer:
                layername = module.__class__.__name__
                if layername == layer_:
                    constant_init(module, self.val, self.bias)


@INITIALIZERS.register_module(name='Xavier')
class XavierInit(BaseInit):
    r"""Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010).

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float | dict): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain=1, bias=0, distribution='normal', layer=None):
        super().__init__(bias, layer)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module):
        if self.layer is None:
            xavier_init(module, self.gain, self.bias, self.distribution)
        else:
            for layer_ in self.layer:
                layername = module.__class__.__name__
                if layername == layer_:
                    xavier_init(module, self.gain, self.bias,
                                self.distribution)


@INITIALIZERS.register_module(name='Normal')
class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float | dict): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self, mean=0, std=1, bias=0, layer=None):
        super().__init__(bias, layer)
        self.mean = mean
        self.std = std

    def __call__(self, module):
        if self.layer is None:
            normal_init(module, self.mean, self.std, self.bias)
        else:
            for layer_ in self.layer:
                layername = module.__class__.__name__
                if layername == layer_:
                    normal_init(module, self.mean, self.std, self.bias)


@INITIALIZERS.register_module(name='Uniform')
class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        a (int | float): the lower bound of the uniform distribution.
            Defaults to 0.
        b (int | float): the upper bound of the uniform distribution.
            Defaults to 1.
        bias (int | float | dict): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, a=0, b=1, bias=0, layer=None):
        super().__init__(bias, layer)
        self.a = a
        self.b = b

    def __call__(self, module):
        if self.layer is None:
            uniform_init(module, self.a, self.b, self.bias)
        else:
            for layer_ in self.layer:
                layername = module.__class__.__name__
                if layername == layer_:
                    uniform_init(module, self.a, self.b, self.bias)


@INITIALIZERS.register_module(name='Kaiming')
class KaimingInit(BaseInit):
    r"""Initialize module paramters with the valuse according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_

    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float | dict): the value to fill the bias or define
            initialization type for bias. Defaults to 0.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal',
                 layer=None):
        super().__init__(bias, layer)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):
        if self.layer is None:
            kaiming_init(module, self.a, self.mode, self.nonlinearity,
                         self.bias, self.distribution)
        else:
            if isinstance(self.layer, str):
                layername = module.__class__.__name__
                if layername == self.layer:
                    kaiming_init(module, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)
            else:
                for layer_ in self.layer:
                    layername = module.__class__.__name__
                    if layername == layer_:
                        kaiming_init(module, self.a, self.mode,
                                     self.nonlinearity, self.bias,
                                     self.distribution)


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


def _initialize_case(module, case):
    if not isinstance(case, (dict, list)):
        raise TypeError(f'case must be a dict or list, but got {type(case)}')

    if isinstance(case, dict):
        case = [case]

    for case_ in case:
        name = case_.pop('name', None)
        if hasattr(module, name):
            _initialize(getattr(module, name), case_)
        else:
            raise RuntimeError(f'module did not have attribute {name}')


def initialize(module, init_cfg):
    """Initialize a module.

    Args:
        module (``torch.nn.Module``): the module will be initialized.
        init_cfg (dict | list[dict]): initialization configuration dict to
            define initializer. OpenMMLab has implemented 7 initializers
            including ``Constant``, ``Xavier``, ``Normal``, ``Uniform``,
            ``Kaiming``, ``Pretrained`` and ``BiasProb`` for bias
            initialization.

    Example:
        >>> module = nn.Linear(2, 3, bias=True)
        >>> init_cfg = dict(type='Constant', val =1 , bias =2)
        >>> initialize(module, init_cfg)

        >>> module = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))
        >>> # define key ``'layer'`` for initializing layer with different
        >>> # configuration
        >>> init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
                dict(type='Constant', layer='Linear', val=2)]
        >>> initialize(module, init_cfg)

        >>> # Omitting ``'layer'`` initialize module with same configuration
        >>> init_cfg = dict(type='Constant', val=1, bias=2)
        >>> initialize(module, init_cfg)

        >>> # define key``'case'`` to initialize some specific case in module
        >>> class FooNet(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.feat = nn.Conv2d(3, 16, 3)
        >>>         self.reg = nn.Conv2d(16, 10, 3)
        >>>         self.cls = nn.Conv2d(16, 5, 3)
        >>> model = FooNet()
        >>> init_cfg = dict(type='Constant', val=1, bias=2,
        >>>     case=dict(type='Constant', name='reg', val=3, bias=4))
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
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        case = cfg.pop('case', None)
        _initialize(module, cfg)

        if case is not None:
            _initialize_case(module, case)
        else:
            # All attributes in module have same initialization.
            pass
