import os.path as osp
import pkgutil
import time
from collections import OrderedDict
from importlib import import_module

import mmcv
import torch
from torch.utils import model_zoo


open_mmlab_model_urls = {
    'vgg16_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/vgg16_caffe-292e1171.pth',  # noqa: E501
    'resnet50_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_caffe-788b5fa3.pth',  # noqa: E501
    'resnet101_caffe': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_caffe-3ad79236.pth',  # noqa: E501
    'resnext50_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50-32x4d-0ab1a123.pth',  # noqa: E501
    'resnext101_32x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d-a5af3160.pth',  # noqa: E501
    'resnext101_64x4d': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth',  # noqa: E501
    'contrib/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_thangvubk-ad1730dd.pth',  # noqa: E501
    'detectron/resnet50_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn-9186a21c.pth',  # noqa: E501
    'detectron/resnet101_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn-cac0ab98.pth',  # noqa: E501
    'jhu/resnet50_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet50_gn_ws-15beedd8.pth',  # noqa: E501
    'jhu/resnet101_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnet101_gn_ws-3e3c308c.pth',  # noqa: E501
    'jhu/resnext50_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn_ws-0d87ac85.pth',  # noqa: E501
    'jhu/resnext101_32x4d_gn_ws': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn_ws-34ac1a9e.pth',  # noqa: E501
    'jhu/resnext50_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext50_32x4d_gn-c7e8b754.pth',  # noqa: E501
    'jhu/resnext101_32x4d_gn': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_32x4d_gn-ac3bb84e.pth',  # noqa: E501
}  # yapf: disable


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            unexpected_keys.append(name)
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        try:
            own_state[name].copy_(param)
        except Exception:
            raise RuntimeError('While copying the parameter named {}, '
                               'whose dimensions in the model are {} and '
                               'whose dimensions in the checkpoint are {}.'
                               .format(name, own_state[name].size(),
                                       param.size()))
    missing_keys = set(own_state.keys()) - set(state_dict.keys())

    err_msg = []
    if unexpected_keys:
        err_msg.append('unexpected key in source state_dict: {}\n'.format(
            ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append('missing keys in source state_dict: {}\n'.format(
            ', '.join(missing_keys)))
    err_msg = '\n'.join(err_msg)
    if err_msg:
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warn(err_msg)
        else:
            print(err_msg)


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        import torchvision
        model_urls = dict()
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if not ispkg:
                _zoo = import_module('torchvision.models.{}'.format(name))
                if hasattr(_zoo, 'model_urls'):
                    _urls = getattr(_zoo, 'model_urls')
                    model_urls.update(_urls)
        model_name = filename[11:]
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        checkpoint = model_zoo.load_url(open_mmlab_model_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = model_zoo.load_url(filename)
    else:
        if not osp.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
    # load state_dict
    if hasattr(model, 'module'):
        load_state_dict(model.module, state_dict, strict, logger)
    else:
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError('meta must be a dict or None, but got {}'.format(
            type(meta)))
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    mmcv.mkdir_or_exist(osp.dirname(filename))
    if hasattr(model, 'module'):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()

    torch.save(checkpoint, filename)
