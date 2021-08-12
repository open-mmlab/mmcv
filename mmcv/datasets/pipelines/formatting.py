from collections.abc import Sequence
from typing import Callable, Dict

import numpy as np
import torch

import mmcv
from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


def apply(container: Dict, key: str, operator: Callable):
    """Apply operator to key-value in a container dict.

    Args:
        container (Dict): A nested dict.
        key (str): A sequence of keys joined by `.`
        operator (Callable): The operator that applied to the value of key.
    """
    nested_keys = key.split('.')
    final_key = nested_keys.pop(-1)
    for key in nested_keys:
        container = container[key]
    container[final_key] = operator(container[final_key])


@PIPELINES.register_module()
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.

    Notes:
        Each key in keys can be a sequence of keys joined by `.`.
        For example, 'ann.gt_label' means results['ann']['gt_label'].
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            apply(results, key, to_tensor)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
