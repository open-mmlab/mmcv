# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import torch
import torch.nn as nn
import popart
import poptorch

from mmcv.utils import Registry
from .model_converter import TrainEvalModel


def build_from_cfg_with_wrapper(
        cfg,
        registry,
        wrapper_func=None,
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

    if wrapper_func is None:
        wrapped_obj_cls = obj_cls
    else:
        wrapped_obj_cls = wrapper_func(obj_cls)
    try:
        return wrapped_obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{wrapped_obj_cls.__name__}: {e}')


def _options_assigner(cfg, options_node):
    # set popart.options by config
    # cfg: dict, python data type
    # options_node: python module or function
    if isinstance(cfg, dict):
        for key in cfg:
            _options_assigner(cfg[key], getattr(options_node, key))
    elif isinstance(cfg, (int, float, str, list)):
        if callable(options_node):
            options_node(cfg)
        else:
            error_msg = f'opts_node type {type(options_node)} not supported'
            raise NotImplementedError(error_msg)
    else:
        error_msg = f'cfg type {type(cfg)} not supported'
        raise NotImplementedError(error_msg)


def parse_ipu_options(ipu_options):
    """parse dictionary to ipu options

    Args:
        ipu_options (dict): A dictionary of ipu settings

    Returns:
        dict[str, poptorch.Options]: Training options and inference options
            of IPU.
    """
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

    # TODO configure these codes
    opts['training']._Popart.set('disableGradAccumulationTensorStreams', True)
    opts['training']._Popart.set(
        'accumulateOuterFragmentSettings.schedule',
        int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts['training'].Precision.enableStochasticRounding(True)

    return opts


def _parse_ipu_options(ipu_options):
    # If it cannot be directly assigned, use if statement to parse it,
    # and if it can be directly assigned, use _options_assigner to assign
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

    _options_assigner(ipu_options, opts)
    return opts


def ipu_model_wrapper(
        model,
        opts,
        optimizer=None,
        logger=None,
        modules_to_record=(),
        ipu_model_cfg=None,
        fp16_cfg=None
        ):
    """Convert torch model to IPU model.

    Args:
        model (nn.Module): The target model to be converted.
        opts (dict[str, poptorch.Options]): IPU options, generated
            by :func:`parse_ipu_options`.
        optimizer (:obj:`torch.optim.Optimizer`, optional): torch
            optimizer, necessary if in training mode
        logger: a logger
        modules_to_record (mmcv.Config, list): Index or name of modules which
            will be recorded for output. It is necessary to specify output for
            static graph of model training or inference.
        ipu_model_cfg (dict): A dictionary contains train_split_edges and
            train_ckpt_nodes, See details in :func:`model_sharding` and
            :func:`recomputation_checkpoint` functions.
        fp16_cfg (dict): Config for IPU fp16 training. Currently supports
            configs: `loss_scale`, `velocity_accum_type` and `accum_type`.
            See details in
            https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/index.html

    Returns:
        TrainEvalModel: IPU wrapped model.
    """
    ipu_model_cfg = ipu_model_cfg or {}
    training = model.training if optimizer is not None else False
    # set mixed-precision
    if fp16_cfg is not None:
        from mmcv.runner.fp16_utils import wrap_fp16_model
        loss_scale = fp16_cfg['loss_scale']
        wrap_fp16_model(model)
        model.half()
        # TODO tmp ussage to set loss scaling for torch original optimizer
        if optimizer is not None:
            optimizer.loss_scaling = loss_scale
            if fp16_cfg.get('velocity_accum_type', False):
                if fp16_cfg['velocity_accum_type'] == 'half':
                    optimizer.velocity_accum_type = torch.half
                else:
                    optimizer.velocity_accum_type = torch.float32
            if fp16_cfg.get('accum_type', False):
                if fp16_cfg['accum_type'] == 'half':
                    optimizer.accum_type = torch.half
                else:
                    optimizer.accum_type = torch.float32
        # TODO support feature alignment for fp16
        if len(modules_to_record) > 0:
            raise NotImplementedError(
                'Feature alignment for fp16 is not implemented')

    # set model partition
    if optimizer is None:
        train_model = None
    else:
        # split model into multi-IPUs if specified
        train_model = model_sharding(
            copy.copy(model).train(),
            ipu_model_cfg.get('train_split_edges', []))

        recomputation_checkpoint(
            train_model,
            ipu_model_cfg.get('train_ckpt_nodes', []))

        # TODO support feature alignment for gradient accumulation mode
        if getattr(opts['training'].Training, 'gradient_accumulation', 1) > 1:
            assert len(modules_to_record) == 0, \
                'Feature alignment for grad-accumulation mode not implemented'

        # TODO support feature alignment for multi-replica mode
        if getattr(opts['training'], 'replication_factor', 1) > 1:
            assert len(modules_to_record) == 0, \
                'Feature alignment for multi-replica mode not implemented'

    # TODO supports different model partitions between train and eval mode
    assert len(ipu_model_cfg.get('eval_split_edges', [])) == 0,\
        'Currently, BeginBlock can only be used once on the same model'
    eval_model = copy.copy(model).eval()

    # wrap model for compilation
    model = TrainEvalModel(train_model, eval_model, options=opts,
                           optimizer=optimizer, logger=logger,
                           modules_to_record=modules_to_record)
    model.train(training)
    return model


def model_sharding(model, split_edges):
    """split models in-place into multi-IPUs.

    Args:
        model (nn.Module): The target model to be split.
        split_edges (list of dict): Model layer names or layer numbers
            of split edge. Each item of ``split_edges`` is a dictionary,
            which may contain the following key-pairs:

            - layer_to_call: PyTorch module to assign to the block
            - user_id (optional): A user defined identifier for the block.
            - ipu_id: The id of the IPU to run on.

        Examples:
            >>> split_edges = [
            ...     dict(layer_to_call='model.conv1', ipu_id=0),
            ...     dict(layer_to_call='model.conv3', ipu_id=1)]
            >>> sharding_model = model_sharding(torch_model, split_edges)

    Returns:
        nn.Module: Split model.
    """
    if len(split_edges) == 0:
        return model
    assert isinstance(split_edges, list)
    spilt_edges_dict = {edge['layer_to_call']: edge for edge in split_edges}

    for idx, (name, module) in enumerate(model.named_modules()):
        if idx in spilt_edges_dict and name in spilt_edges_dict:
            raise ValueError(
                'The same layer is referenced twice while doing model'
                f' partition: idx is {idx} and name is {name}')

        edge = spilt_edges_dict.pop(name, None)
        edge = spilt_edges_dict.pop(idx, edge)
        if edge is not None:
            poptorch.BeginBlock(module, edge.get(
                'user_id', name), edge['ipu_id'])

    # ensure all split_edges are used
    if len(spilt_edges_dict) > 0:
        split_edge_names = list(spilt_edges_dict.keys())
        raise RuntimeError(
            f'split_edges: {split_edge_names} are not contained in the model')
    return model


def recomputation_checkpoint(model: nn.Module, module_names: list):
    """Annotates the output of a module to be checkpointed instead of
    recomputed.

    If recomputation mode is enabled, ipu will release the activations of
    the middle layers to save memory. During the backward of gradient,
    the activation of the middle layer will be recalculated again.
    This function is used to declare the activations of some intermediate
    layers that need to be saved in order to skip the recomputation of
    some layers.

    Args:
        model (nn.Module): The target model to apply recomputation
            checkpoint.
        module_names (list): Layer names of module.
    """
    def recompute_outputs(module, inputs, outputs):
        if type(outputs) is tuple:
            return tuple(poptorch.recomputationCheckpoint(y) for y in outputs)
        else:
            return poptorch.recomputationCheckpoint(outputs)

    for name, module in model.named_modules():
        if name in module_names:
            module.register_forward_hook(recompute_outputs)
            module_names.remove(name)

    # check all module_names are used
    assert len(module_names) == 0,\
        f'recomputed nodes: {module_names} are not contained in the model'
