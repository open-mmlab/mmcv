# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from abc import ABCMeta, abstractmethod
from torch.utils.data import RandomSampler
try:
    import poptorch
    from .model_converter import trainingModel
    IPU_MODE = True
except ImportError:
    IPU_MODE = False

from ..builder import RUNNERS
from ..hooks import HOOKS, LrUpdaterHook
from ...utils import Registry


def build_from_cfg_with_wrapper(cfg, registry, wrapper_func, default_args=None):
    """Build a module from config dict and wrap module with "wrapper_func"

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
        wrapper_func (function): Used to wrap class

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be an mmcv.Registry object, '
                        f'but got {type(registry)}')
    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    wrapped_obj_cls = wrapper_func(obj_cls)
    try:
        return wrapped_obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{wrapped_obj_cls.__name__}: {e}')


def _opts_assigner(_cfg, opts_node):
    # _cfg: dict, python data type
    # opts_node: python module or function
    if isinstance(_cfg, dict):
        for _key in _cfg:
            _opts_assigner(_cfg[_key],getattr(opts_node,_key))
    elif isinstance(_cfg, (int,float,str)):
        if callable(opts_node):
            opts_node(_cfg)
        else:
            error_msg = 'opts_node type {} not supported'.format(type(opts_node))
            raise NotImplementedError(error_msg)      
    else:
        error_msg = 'cfg type {} not supported'.format(type(_cfg))
        raise NotImplementedError(error_msg)


def parse_ipu_options(ipu_options):
    opts = poptorch.Options()
    _opts_assigner(ipu_options, opts)
    return opts


def wrap_model(model, opts, optimizer):
    # three things need to do
    # wrap model with poptorch
    # set mixed-precision
    # set model partition
    model = trainingModel(model, options=opts, optimizer=optimizer)
    # TODO set mixed-precision
    # TODO set model partition
    return model


def wrap_data_loader(data_loader, opts):
    # get all params need to initialize ipu dataloader
    dataset = data_loader.dataset
    batch_size = data_loader.batch_size
    num_workers = data_loader.num_workers
    # dist = data_loader.dist
    worker_init_fn = data_loader.worker_init_fn
    sampler = data_loader.sampler
    shuffle = isinstance(data_loader.sampler,RandomSampler)
    collate_fn = data_loader.collate_fn
    pin_memory = data_loader.pin_memory
    # TODO maybe need to do some changes to data_loader
    return data_loader


def wrap_lr_update_hook(lr_hook_class,):
    assert issubclass(lr_hook_class, LrUpdaterHook)
    class ipu_lr_hook_class(lr_hook_class):
        def _set_lr(self, runner, *args, **kwargs):
            result = super()._set_lr(runner, *args, **kwargs)
            assert result is None # _set_lr should return nothing
            runner.model.setOptimizer(runner.optimizer)
    return ipu_lr_hook_class

