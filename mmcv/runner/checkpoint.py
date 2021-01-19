# Copyright (c) Open-MMLab. All rights reserved.
import io
import os
import os.path as osp
import pkgutil
import time
import warnings
from collections import OrderedDict
from importlib import import_module
from tempfile import TemporaryDirectory

import torch
import torchvision
from torch.optim import Optimizer
from torch.utils import model_zoo

import mmcv
from ..fileio import FileClient
from ..fileio import load as load_file
from ..parallel import is_module_wrapper
from ..utils import mkdir_or_exist
from .dist_utils import get_dist_info

ENV_MMCV_HOME = 'MMCV_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_mmcv_home():
    mmcv_home = os.path.expanduser(
        os.getenv(
            ENV_MMCV_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmcv')))

    mkdir_or_exist(mmcv_home)
    return mmcv_home


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
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def load_url_dist(url, model_dir=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = model_zoo.load_url(url, model_dir=model_dir)
    return checkpoint


def load_pavimodel_dist(model_path, map_location=None):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    if rank == 0:
        model = modelcloud.get(model_path)
        with TemporaryDirectory() as tmp_dir:
            downloaded_file = osp.join(tmp_dir, model.name)
            model.download(downloaded_file)
            checkpoint = torch.load(downloaded_file, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            model = modelcloud.get(model_path)
            with TemporaryDirectory() as tmp_dir:
                downloaded_file = osp.join(tmp_dir, model.name)
                model.download(downloaded_file)
                checkpoint = torch.load(
                    downloaded_file, map_location=map_location)
    return checkpoint


def load_fileclient_dist(filename, backend, map_location):
    """In distributed setting, this function only download checkpoint at local
    rank 0."""
    rank, world_size = get_dist_info()
    rank = int(os.environ.get('LOCAL_RANK', rank))
    allowed_backends = ['ceph']
    if backend not in allowed_backends:
        raise ValueError(f'Load from Backend {backend} is not supported.')
    if rank == 0:
        fileclient = FileClient(backend=backend)
        buffer = io.BytesIO(fileclient.get(filename))
        checkpoint = torch.load(buffer, map_location=map_location)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            fileclient = FileClient(backend=backend)
            buffer = io.BytesIO(fileclient.get(filename))
            checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


def get_torchvision_models():
    model_urls = dict()
    for _, name, ispkg in pkgutil.walk_packages(torchvision.models.__path__):
        if ispkg:
            continue
        _zoo = import_module(f'torchvision.models.{name}')
        if hasattr(_zoo, 'model_urls'):
            _urls = getattr(_zoo, 'model_urls')
            model_urls.update(_urls)
    return model_urls


def get_external_models():
    mmcv_home = _get_mmcv_home()
    default_json_path = osp.join(mmcv.__path__[0], 'model_zoo/open_mmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmcv_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmcv.__path__[0], 'model_zoo/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmcv.__path__[0],
                                   'model_zoo/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def _process_mmcls_checkpoint(checkpoint):
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


class BaseCheckpointLoader:
    """Abstract class of checkpoint loader.

    All loaders need to implement the classmethod api: ``load_checkpoint()``.
    it's load checkpoint by filename and map_location parameter
    """

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        raise NotImplementedError()


class NativeCheckpointLoader(BaseCheckpointLoader):
    """Native checkpoint loader."""

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        """load checkpoint by local file path.

        Args:
            filename (str): local checkpoint file path
            map_location (str): Same as :func:`torch.load`.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        if not osp.isfile(filename):
            raise IOError(f'{filename} is not a checkpoint file')
        checkpoint = torch.load(filename, map_location=map_location)
        return checkpoint


class CheckpointLoaderClient:
    """A general checkpoint loader client to load checkpoint.

    Attributes:
        _loader (dict[loader]): .
        _native_checkpoint_handler (:obj:`NativeCheckpointLoader`):
           The default loader object.
    """

    _loader = {}
    _native_checkpoint_loader = NativeCheckpointLoader

    @classmethod
    def _register_loader(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._loader) or force:
                cls._loader[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a storage backend, '
                    'add "force=True" if you want to override it')
        # sort
        cls._loader = OrderedDict(
            sorted(cls._loader.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_loader(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoaderClient.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (class, optional): The loader class to be registered,
                which must be a subclass of :class:`BaseCheckpointLoader`.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_loader(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_loader(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the
        native loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            loader (Loader)
        """

        for p in cls._loader:
            if path.startswith(p):
                return cls._loader[p]
        return cls._native_checkpoint_loader

    @classmethod
    def load_checkpoint(cls, filepath, map_location=None, logger=None):
        """load checkpoint through URL scheme file path.

        Args:
            filename (str): checkpoint file path with given prefix
            map_location (str or None): Same as :func:`torch.load`.
            logger (:mod:`logging.Logger` or None): The logger for message.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filepath)
        if isinstance(checkpoint_loader, type):
            # class object
            class_name = checkpoint_loader.__name__
        else:
            # class instance object
            class_name = checkpoint_loader.__class__.__name__
        mmcv.print_log(f'Use {class_name} loader', logger)
        return checkpoint_loader.load_checkpoint(filepath, map_location)


@CheckpointLoaderClient.register_loader(prefixes=('http://', 'https://'))
class HTTPURLLoadCheckpointLoader(BaseCheckpointLoader):
    """HTTP URL scheme checkpoint loader."""

    @classmethod
    def load_checkpoint(cls, filename, map_location=None):
        """load checkpoint through HTTP or HTTPS scheme path.

        Args:
            filename (str): checkpoint file path with modelzoo or
                torchvision prefix
            map_location (None): it's not use.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint = load_url_dist(filename)
        return checkpoint


@CheckpointLoaderClient.register_loader(
    prefixes=('modelzoo://', 'torchvision://'))
class TorchLoadCheckpointLoader(BaseCheckpointLoader):
    """Checkpoint loader that can process modelzoo or torchvision file path
    prefix."""

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        """load checkpoint through the file path prefixed with modelzoo or
        torchvision.

        Args:
            filename (str): checkpoint file path with modelzoo or
                torchvision prefix
            map_location (str): Same as :func:`torch.load`.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """
        model_urls = get_torchvision_models()
        if filename.startswith('modelzoo://'):
            warnings.warn(
                'The URL scheme of "modelzoo://" is deprecated, please '
                'use "torchvision://" instead')
            model_name = filename[11:]
        else:
            model_name = filename[14:]
        checkpoint = load_url_dist(model_urls[model_name])
        return checkpoint


@CheckpointLoaderClient.register_loader(prefixes='open-mmlab://')
class OpenMMLabCheckpointLoader(BaseCheckpointLoader):
    """Checkpoint loader that can process open-mmlab file path prefix."""

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        """load checkpoint through the file path prefixed with open-mmlab.

        Args:
            filename (str): checkpoint file path with open-mmlab prefix
            map_location (str): Same as :func:`torch.load`.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        model_urls = get_external_models()
        model_name = filename[13:]
        deprecated_urls = get_deprecated_model_names()
        if model_name in deprecated_urls:
            warnings.warn(f'open-mmlab://{model_name} is deprecated in favor '
                          f'of open-mmlab://{deprecated_urls[model_name]}')
            model_name = deprecated_urls[model_name]
        model_url = model_urls[model_name]
        # check if is url
        if model_url.startswith(('http://', 'https://')):
            checkpoint = load_url_dist(model_url)
        else:
            filename = osp.join(_get_mmcv_home(), model_url)
            if not osp.isfile(filename):
                raise IOError(f'{filename} is not a checkpoint file')
            checkpoint = torch.load(filename, map_location=map_location)
        return checkpoint


@CheckpointLoaderClient.register_loader(prefixes='mmcls://')
class MMCLSCheckpointLoader(BaseCheckpointLoader):
    """Checkpoint loader that can process mmcls file path prefix."""

    @classmethod
    def load_checkpoint(cls, filename, map_location=None):
        """load checkpoint through the file path prefixed with mmcls.

        Args:
            filename (str): checkpoint file path with mmcls prefix
            map_location (None): it's not use.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        model_urls = get_mmcls_models()
        model_name = filename[8:]
        checkpoint = load_url_dist(model_urls[model_name])
        checkpoint = _process_mmcls_checkpoint(checkpoint)
        return checkpoint


@CheckpointLoaderClient.register_loader(prefixes='pavi://')
class PAVICheckpointLoader(BaseCheckpointLoader):
    """Checkpoint loader that can process pavi file path prefix."""

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        """load checkpoint through the file path prefixed with pavi.

        Args:
            filename (str): checkpoint file path with pavi prefix
            map_location (str): Same as :func:`torch.load`.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        model_path = filename[7:]
        checkpoint = load_pavimodel_dist(model_path, map_location=map_location)
        return checkpoint


@CheckpointLoaderClient.register_loader(prefixes='s3://')
class S3CheckpointLoader(BaseCheckpointLoader):
    """Checkpoint loader that can process s3 file path prefix."""

    @classmethod
    def load_checkpoint(cls, filename, map_location):
        """load checkpoint through the file path prefixed with s3.

        Args:
            filename (str): checkpoint file path with s3 prefix
            map_location (str): Same as :func:`torch.load`.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint = load_fileclient_dist(
            filename, backend='ceph', map_location=map_location)
        return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = CheckpointLoaderClient.load_checkpoint(
        filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # load state_dict
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


def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.

    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Default: False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


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
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }
    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
