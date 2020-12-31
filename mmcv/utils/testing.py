# Copyright (c) Open-MMLab.


def check_dict(result_dict, key_list, value_list):
    """Check if the result_dict is correct. The function does not support the
    case that both values in result_dict and value_list are nonstandard python
    data types.

    Args:
        result_dict(dict): Dict to be checked.
        key_list(tuple): Tuple of checked keys.
        value_list(tuple): Tuple of target values.

    Returns:
        bool: Whether the result_dict is correct.
    """
    for key, value in zip(key_list, value_list):
        if result_dict[key] != value:
            return False
    return True


def check_class_attr(obj, attr_list, value_list):
    """Check if attribute of class object is correct.

    Args:
        obj(object): Class object to be checked.
        attr_list(tuple[str]): Tuple of inner attribute names ot be checked.
        value_list(tuple): Tuple of target values.

    Returns:
        bool: Whether the attribute of class object is correct.
    """
    for attr, value in zip(attr_list, value_list):
        if not hasattr(obj, attr) or getattr(obj, attr) != value:
            return False
    return True


def assert_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def assert_keys_equal(result_keys, target_keys):
    """Check if target_keys is equal to result_keys."""
    return set(result_keys) == set(target_keys)


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    from .parrots_wrapper import _BatchNorm
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def is_block(module, block_candidates):
    """Check if the module is the specified block.

    Args:
        module(nn.Module): The module to be checked.
        block_candidates(tuple[nn.module]): Tuple of block candidates.

    Returns:
        bool: Whether the module is the specified block.
    """
    return isinstance(module, block_candidates)


def is_norm(module):
    """Check if the module is a norm layer."""
    from .parrots_wrapper import _BatchNorm, _InstanceNorm
    from torch.nn import GroupNorm, LayerNorm
    norm_layer_candidates = (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm)
    return is_block(module, norm_layer_candidates)


def is_all_zeros(module):
    """Check if the weight (and bias) of the module is all zero."""
    weight_data = module.weight.data
    is_weight_zero = weight_data.allclose(
        weight_data.new_zeros(weight_data.size()))

    if hasattr(module, 'bias') and module.bias is not None:
        bias_data = module.bias.data
        is_bias_zero = bias_data.allclose(
            bias_data.new_zeros(bias_data.size()))
    else:
        is_bias_zero = True

    return is_weight_zero and is_bias_zero
