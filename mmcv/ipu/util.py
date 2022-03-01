# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import torch
import poptorch
from mmcv.runner.fp16_utils import wrap_fp16_model
from ..utils import Registry
from .model_converter import TrainEvalModel


def build_from_cfg_with_wrapper(
        cfg,
        registry,
        wrapper_func,
        default_args=None):
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
    # set popart.options by config
    # _cfg: dict, python data type
    # opts_node: python module or function
    if isinstance(_cfg, dict):
        for _key in _cfg:
            _opts_assigner(_cfg[_key], getattr(opts_node, _key))
    elif isinstance(_cfg, (int, float, str, list)):
        if callable(opts_node):
            opts_node(_cfg)
        else:
            error_msg = 'opts_node type {} not supported'.format(
                type(opts_node))
            raise NotImplementedError(error_msg)
    else:
        error_msg = 'cfg type {} not supported'.format(type(_cfg))
        raise NotImplementedError(error_msg)


def parse_ipu_options(ipu_options):
    # set ipu options for inference and training by config
    train_cfgs = ipu_options.pop('train_cfgs', {})
    eval_cfgs = ipu_options.pop('eval_cfgs', {})
    eval_cfgs['replicationFactor'] = 1  # eval mode only use one replica
    eval_cfgs['executionStrategy'] = 'ShardedExecution'
    # overwrite default ipu options with specified train cfgs
    training_ipu_options = {**ipu_options, **train_cfgs}
    # overwrite default ipu options with specified eval cfgs
    inference_ipu_options = {**ipu_options, **eval_cfgs}

    opts = {'training': _parse_ipu_options(training_ipu_options),
            'inference': _parse_ipu_options(inference_ipu_options)}
    return opts


def _parse_ipu_options(ipu_options):
    # If it cannot be directly assigned, use if statement to parse it,
    # and if it can be directly assigned, use _opts_assigner to assign
    opts = poptorch.Options()
    if 'availableMemoryProportion' in ipu_options:
        availableMemoryProportion = ipu_options.pop(
            'availableMemoryProportion')
        mem_prop = {f'IPU{i}': availableMemoryProportion[i]
                    for i in range(len(availableMemoryProportion))}
        opts.setAvailableMemoryProportion(mem_prop)
    if 'executionStrategy' in ipu_options:
        executionStrategy = ipu_options.pop('executionStrategy')
        if executionStrategy == 'SameAsIpu':
            opts.setExecutionStrategy(poptorch.PipelinedExecution(
                getattr(poptorch.AutoStage, executionStrategy)))
        elif executionStrategy == 'ShardedExecution':
            opts.setExecutionStrategy(poptorch.ShardedExecution())
        else:
            raise NotImplementedError
    if 'partialsType' in ipu_options:
        partialsType = ipu_options.pop('partialsType')
        opts.Precision.setPartialsType(
            getattr(torch, partialsType))  # half or float
    _opts_assigner(ipu_options, opts)
    return opts


def ipu_model_wrapper(
        model,
        opts,
        optimizer=None,
        logger=None,
        modules_to_record=[],
        pipeline_cfg={},
        fp16_cfg=None
        ):
    # TrainEvalModel will shallow copy the model,
    # so any changes to the model must be placed before TrainEvalModel
    # set mixed-precision
    if fp16_cfg is not None:
        loss_scale = fp16_cfg['loss_scale']
        wrap_fp16_model(model)
        model.half()
        # TODO tmp ussage to set loss scaling for torch original optimzier
        optimizer.loss_scaling = loss_scale

    # set model partition
    if optimizer is None:
        train_model = None
    else:
        # split model into multi-ipus if specified
        train_model = model_sharding(copy.copy(model).train(),
                                     pipeline_cfg.get('train_split_edges', []))
    # split model into multi-ipus if specified
    eval_model = model_sharding(copy.copy(model).eval(), pipeline_cfg.get(
        'eval_split_edges', []))

    # wrap model for compilation
    model = TrainEvalModel(train_model, eval_model, options=opts,
                           optimizer=optimizer, logger=logger,
                           modules_to_record=modules_to_record)

    return model


# def wrap_data_loader(data_loader, opts):
#     # get all params need to initialize ipu dataloader
#     dataset = data_loader.dataset
#     batch_size = data_loader.batch_size
#     num_workers = data_loader.num_workers
#     # dist = data_loader.dist
#     worker_init_fn = data_loader.worker_init_fn
#     sampler = data_loader.sampler
#     shuffle = isinstance(data_loader.sampler,RandomSampler)
#     collate_fn = data_loader.collate_fn
#     pin_memory = data_loader.pin_memory
#     # TODO maybe need to do some changes to data_loader
#     return data_loader


def model_sharding(model, split_edges):
    # shard model into multi-ipus according to the pipeline config
    # three args needed in pipeline_cfg['split_edges']:
    """
    :param layer_to_call: model layer name or layer number
    :param user_id: A user defined identifier for the block.
        Blocks with the same id are considered as being a single block.
        Block identifiers are also used to manually specify pipelines or
        phases.
    :param ipu_id: The id of the IPU to run on.
                    Note that the ``ipu_id`` is an index
                    in a multi-IPU device within PopTorch, and is
                    separate and distinct from the device ids used by
                    ``gc-info``.
    """
    if len(split_edges) == 0:
        return model
    assert isinstance(split_edges, list)
    spilt_edges_dic = {ele['layer_to_call']: ele for ele in split_edges}
    for idx, (_name, _module) in enumerate(model.named_modules()):
        assert not (idx in spilt_edges_dic and _name in spilt_edges_dic),\
            "The same layer is referenced twice while doing model partition: "\
            "idx is {} and name is {}".format(idx, _name)
        edge = spilt_edges_dic.pop(_name, None)
        edge = spilt_edges_dic.pop(idx, edge)
        if edge is not None:
            poptorch.BeginBlock(_module, edge.get(
                'user_id', _name), edge['ipu_id'])
    # check all split_edges are used
    assert len(spilt_edges_dic) == 0,\
        'split_edges: {} are not contained in the model'.format(
                                        list(spilt_edges_dic.keys()))
    return model
