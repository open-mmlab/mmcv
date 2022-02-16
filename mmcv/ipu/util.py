# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import torch
import poptorch
from abc import ABCMeta, abstractmethod
from torch.utils.data import RandomSampler
from mmcv.runner.fp16_utils import wrap_fp16_model
from ..utils import Registry
from .model_converter import TrainEvalModel


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
    elif isinstance(_cfg, (int,float,str,list)):
        if callable(opts_node):
            opts_node(_cfg)
        else:
            error_msg = 'opts_node type {} not supported'.format(type(opts_node))
            raise NotImplementedError(error_msg)      
    else:
        error_msg = 'cfg type {} not supported'.format(type(_cfg))
        raise NotImplementedError(error_msg)


def parse_ipu_options(ipu_options):
    training_ipu_options = ipu_options
    inference_ipu_options = copy.deepcopy(ipu_options)
    if 'Inference' in training_ipu_options: training_ipu_options.pop('Inference')
    if 'Training' in inference_ipu_options: inference_ipu_options.pop('Training')
    opts = {'training':_parse_ipu_options(training_ipu_options),
            'inference':_parse_ipu_options(inference_ipu_options)}
    return opts


def _parse_ipu_options(ipu_options):
    opts = poptorch.Options()
    if 'availableMemoryProportion' in ipu_options:
        availableMemoryProportion = ipu_options.pop('availableMemoryProportion')
        mem_prop = {f'IPU{i}': availableMemoryProportion[i] for i in range(len(availableMemoryProportion))}
        opts.setAvailableMemoryProportion(mem_prop)
    if 'executionStrategy' in ipu_options:
        executionStrategy = ipu_options.pop('executionStrategy')
        opts.setExecutionStrategy(poptorch.PipelinedExecution(getattr(poptorch.AutoStage,executionStrategy)))
    if 'partialsType' in ipu_options:
        partialsType = ipu_options.pop('partialsType')
        opts.Precision.setPartialsType(getattr(torch, partialsType)) # half or float
    _opts_assigner(ipu_options, opts)
    return opts


def ipu_model_wrapper(model, opts, optimizer, logger=None, modules_to_record=[], pipeline_cfg={}, fp16_cfg=None):
    # TrainEvalModel will shallow copy the model, so any changes to the model must be placed before TrainEvalModel
    # set mixed-precision
    if fp16_cfg is not None:
        loss_scale = fp16_cfg['loss_scale']
        wrap_fp16_model(model)
        model.half()

    # set model partition
    model = add_split_edges(model, pipeline_cfg) # split model into multi-ipus if specified
    
    # wrap model for compilation
    model = TrainEvalModel(model, options=opts, optimizer=optimizer, logger=logger, modules_to_record=modules_to_record)

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


def add_split_edges(model, pipeline_cfg):
    if len(pipeline_cfg) == 0: return model
    assert isinstance(pipeline_cfg, dict)
    spilt_edges_dic = {ele['layer_to_call']:ele for ele in pipeline_cfg['split_edges']}
    for idx, (_name, _module) in enumerate(model.named_modules()):
        assert not (idx in spilt_edges_dic and _name in spilt_edges_dic), "The same layer is referenced twice while doing model partition: idx is {} and name is {}".format(idx, _name)
        edge = spilt_edges_dic.get(_name, None)
        edge = spilt_edges_dic.get(idx, edge)
        if edge is not None:
            poptorch.BeginBlock(_module, edge.get('user_id',_name), edge['ipu_id'])

    return model