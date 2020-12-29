# Copyright (c) Open-MMLab. All rights reserved.
import logging
import numpy as np
import torch.nn as nn

from mmcv.runner import (_load_checkpoint, _load_checkpoint_with_prefix,
                         load_checkpoint, load_state_dict)
from mmcv.utils import print_log


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
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
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def initialize(module, init_cfg):
    """Initialize a module

    Args:
        module (``torch.nn.Module``): An module will be initialized.
        init_cfg (dict):  Initialization config dict. It should at least
            contain the key "type", and "type" value must be "random" or
            "pretrained", which indicates weights initialization with values
            drawn from random distribution or a pretrained model. The keys in
            init_cfg with "random" type must be "Conv2d", "Linear",
            "BatchNorm2d" etc., names of layers with learnable parameters.
            The init_cfg with "pretrained" type must contain key "checkpoint"
            that indicates where to load the pretrained model.

    Example:
        >>> model = nn.Linear(2, 3, bias=True)
        >>> # the values of "Linear" is dict that refers to the initilization
            # function and argments. OpenMMLab has implemented initialization
            # functions including `constant_init`, `xavier_init`, `normal_init`,
            # `uniform_init`, `kaiming_init`, `caffe2_xavier_init` and
            # `bias_init_wth_prob` in mmcv.cnn.utils.
        >>> init_cfg = dict(type='random', Linear=dict(function='constant_init',
                val=1, bias=2))
        >>> initialize(model, init_cfg)
        >>> for name, param in model.named_parameters():
                print(name, param)
        weight Parameter containing:
        tensor([[1., 1.],
                [1., 1.],
                [1., 1.]], requires_grad=True)
        bias Parameter containing:
        tensor([2., 2., 2.], requires_grad=True)
        >>> # It also supports initialization functions implemented in pytorch.
        >>> init_cfg = dict(type='random',
                Linear=[
                    dict(function='constant', parameters='weight', val=3),
                    dict(function='constant', parameters='bias', val=4)
                ])
        >>> initialize(module, init_cfg)
        >>> for name, param in model.named_parameters():
                print(name, param)
        weight Parameter containing:
        tensor([[3., 3.],
                [3., 3.],
                [3., 3.]], requires_grad=True)
        bias Parameter containing:
        tensor([4., 4., 4.], requires_grad=True)

        >>> from mmdet.models.backbones import ResNet
        >>> model = ResNet(depth=50)
        >>> # Initialize weights with the pretrained model.
        >>> init_cfg = dict(type='pretrained',
                checkpoint='torchvision://resnet50')
        >>> initialize(module, init_cfg)

        >>> # Intialize weights of a sub-module with the specific part of
        >>> # a pretrained model by using "prefix".
        >>> url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
        >>>     'retinanet_r50_fpn_1x_coco/'\
        >>>     'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
        >>> init_cfg = dict(type='pretrained',
                checkpoint=url, prefix='backbone.')

    """
    if not isinstance(init_cfg, dict):
        raise TypeError(f'init_cfg must be a dict, but got {type(init_cfg)}')
    if 'type' not in init_cfg:
        raise TypeError(
            f'init_cfg must contain the key "type", but got {init_cfg}')
    init_type = init_cfg.get('type')
    if init_type == 'pretrained':
        pretrained_init(module, init_cfg)
    elif init_type == 'random':
        random_init(module, init_cfg)
    else:
        raise TypeError(
            'the type of init_cfg must be "random" or "pretrained", '
            f'but got {init_type}')


def random_init(module, init_cfg):
    """Initialize module with values drawn from a random distribution.

    Args:
        module (nn.Module): An module will be initialized.
        init_cfg (dict): Initialization config dict.
    """

    def init_func(m):
        classname = m.__class__.__name__

        if init_cfg.get(classname) is not None:
            if isinstance(init_cfg[classname], list):
                for cfg_ in init_cfg[classname]:
                    args = cfg_.copy()
                    parameters = args.pop('parameters')
                    init_type = args.pop('function')
                    if hasattr(nn.init, init_type + "_"):
                        func = getattr(nn.init, init_type + "_")
                        if getattr(m, parameters) is not None:
                            func(getattr(m, parameters), **args)
                    elif init_type == 'bias_init_with_prob':
                        bias_cls = bias_init_with_prob(**args)
                        nn.init.constant_(getattr(m, parameters), bias_cls)

            else:
                args = init_cfg[classname].copy()
                init_type = args.pop('function')
                func = None
                if hasattr(nn.init, init_type + "_"):
                    func = getattr(nn.init, init_type + "_")
                else:
                    func = eval(init_type)
                func(m, **args)

    module.apply(init_func)


def pretrained_init(module, init_cfg):
    """Initialize module by loading a pretrained model

    Args:
        module (nn.Module): An module will be initialized.
        init_cfg (dict): Initialization config dict.

    """
    if 'checkpoint' not in init_cfg:
        raise TypeError(
            'init_cfg with "pretrained" type must contain the key "checkpoint"'
        )
    checkpoint = init_cfg.get('checkpoint')
    prefix = init_cfg.get('prefix', '')
    logger = logging.getLogger()
    print_log(f'load model from: {checkpoint}', logger=logger)
    if prefix == '':
        load_checkpoint(module, checkpoint, strict=False, logger=logger)

    else:
        state_dict = _load_checkpoint_with_prefix(prefix, checkpoint)
        load_state_dict(module, state_dict, strict=False, logger=logger)
