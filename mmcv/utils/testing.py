# Copyright (c) Open-MMLab.
import torch
from torch.nn import GroupNorm

from mmcv.utils import _BatchNorm


def check_dict(result_dict, key_list, value_list):
    """check if result_dict is correct.

    Args:
        result_dict(dict): dict to be checked.
        key_list(tuple): tuple of checked keys.
        value_list(tuple): tuple of target values.
    """
    for key, value in zip(key_list, value_list):
        if result_dict[key] != value:
            return False
    return True


def check_class_attr(obj, attr_list, value_list):
    """Check if attribute of class object is correct.

    Args:
        obj(object): class object to be checked.
        attr_list(tuple[str]): tuple of inner attribute names ot be checked.
        value_list(tuple): tuple of target values.
    """
    for attr, value in zip(attr_list, value_list):
        if not hasattr(obj, attr) or getattr(obj, attr) != value:
            return False
    return True


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


def check_keys_equal(result_keys, target_keys):
    """check if all elements in target_keys is in reusult_keys."""
    return set(result_keys) == set(target_keys)


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def is_block(module, block_candidates):
    """Check if the module is the specified block.

    Args:
        module(nn.Module): The module to be checked.
        block_candidates(tuple[module]): tuple of block candidates
    """
    if isinstance(module, block_candidates):
        return True
    return False


def is_norm(module):
    """Check if the module is a norm layer."""
    return is_block(module, (GroupNorm, _BatchNorm))


def is_all_zeros(module):
    """Check if the weight (and bias) of the module is all zero."""
    is_weight_zero = torch.allclose(module.weight.data,
                                    torch.zeros_like(module.weight.data))

    if hasattr(module, 'bias'):
        is_bias_zero = torch.allclose(module.bias.data,
                                      torch.zeros_like(module.bias.data))
    else:
        is_bias_zero = True

    return is_weight_zero and is_bias_zero
