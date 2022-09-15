# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union

import mmengine
import numpy as np
import torch

from .base import BaseTransform
from .builder import TRANSFORMS


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmengine.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@TRANSFORMS.register_module()
class ToTensor(BaseTransform):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys

    def transform(self, results: dict) -> dict:
        """Transform function to convert data to `torch.Tensor`.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: `keys` in results will be updated.
        """
        for key in self.keys:

            key_list = key.split('.')
            cur_item = results
            for i in range(len(key_list)):
                if key_list[i] not in cur_item:
                    raise KeyError(f'Can not find key {key}')
                if i == len(key_list) - 1:
                    cur_item[key_list[i]] = to_tensor(cur_item[key_list[i]])
                    break
                cur_item = cur_item[key_list[i]]

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor(BaseTransform):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Required keys:

    - all these keys in `keys`

    Modified Keys:

    - all these keys in `keys`

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys: dict) -> None:
        self.keys = keys

    def transform(self, results: dict) -> dict:
        """Transform function to convert image in results to
        :obj:`torch.Tensor` and transpose the channel order.
        Args:
            results (dict): Result dict contains the image data to convert.
        Returns:
            dict: The result dict contains the image converted
            to :obj:``torch.Tensor`` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = (to_tensor(img.transpose(2, 0, 1))).contiguous()
        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + f'(keys={self.keys})'
