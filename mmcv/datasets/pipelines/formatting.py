# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Sequence, Union

import numpy as np
import torch

import mmcv
from ..builder import PIPELINES


def to_tensor(data: Union[torch.Tensor, np.ndarray, Sequence, int, float]):
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


def apply_operator(container: dict, key: str, operator: Callable):
    """Apply operator to key-value in a container dict.

    Args:
        container (dict): A nested dict.
        key (str): The key of value. It can be a concatenation of multiple
            strings to get value from nested dict, and the separator is ``.``.
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
        keys (Sequence[str]): Keys of values that need to be converted to
            Tensor. Each key in ``keys`` can be a concatenation of multiple
            strings separated by by ``.``.

    Example:
        >>> from mmcv.datasets.pipelines import ToTensor
        >>> transform = ToTensor(keys=['img', 'ann.label'])
        >>> input_dict = {'img': [1, 2], 'ann': {'label': 1}}
        >>> output_dict = transform(input_dict)
        >>> print(output_dict)
        {'img': tensor([1, 2]), 'ann': {'label': tensor([1])}}
    """

    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def __call__(self, results: dict):
        """Call function to convert to tensor.

        Args:
            results (dict): Required keys are all keys in ``self.keys``.

        Returns:
            dict: Output results.

            - Updated keys are all keys in ``self.keys``.
        """
        for key in self.keys:
            apply_operator(results, key, to_tensor)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class ImageToTensor:
    """Convert image to :obj:`torch.Tensor` by given keys.

    Notes:
        The dimension order of input image is (H, W, C). The pipeline will
        convert it to (C, H, W). If only 2 dimension (H, W) is given, the
        output would be (1, H, W).

    Example:
        >>> import numpy as np
        >>> from mmcv.datasets.pipelines import ImageToTensor
        >>> transform = ImageToTensor()
        >>> input_dict = {'img': np.zeros([16, 16, 3]), 'img_fields': ['img']}
        >>> output_dict = transform(input_dict)
        >>> print(output_dict['img'].shape)
        torch.Size([3, 16, 16])
    """

    def __call__(self, results: dict):
        """Call function to convert image to tensor.

        Args:
            results (dict): The required key is ``img_fields``.

        Returns:
            dict: Output results.

            - Updated keys are all keys in ``img_fields``.
        """
        assert 'img_fields' in results, 'ImageToTensor requires key '\
            '"img_fields", Please check your pipelines.'

        for key in results['img_fields']:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__
