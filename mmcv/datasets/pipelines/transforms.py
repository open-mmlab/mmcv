from collections.abc import Sequence
from typing import Optional

import numpy as np

import mmcv
from ..builder import PIPELINES


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    `results` is the input of `__call__()`. The required key of `results` is
    `img_fields`, which is a list of string. Every item of it is also the key
    of `results`, whose value is an image. After invoking the `__call__()`, the
    return results will add an another key `img_norm_cfg`.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.

    Example:
        >>> from mmcv.image.io import imread
        >>> from mmcv.datasets.pipelines import Normalize
        >>> cfg = {
        ...     'mean': [123.675, 116.28, 103.53],
        ...     'std': [58.395, 57.12, 57.375]
        ... }
        >>> img = imread('img_path')  # img_path is the path of image
        >>> results = {
        ...     'img': img,
        ...     'img_fields': ['img']
        ... }
        >>> normalize = Normalize(**cfg)
        >>> results = normalize(results)
    """

    def __init__(self, mean: Sequence, std: Sequence, to_rgb: bool = True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results: dict) -> dict:
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline. Required key is
                'img_fields'.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        assert 'img_fields' in results, \
            '"img_fields" is a required key of results'

        for key in results['img_fields']:
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor".

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.

    Example:
        >>> from mmcv.image.io import imread
        >>> from mmcv.datasets.pipelines import Pad
        >>> img = imread('/path/of/your/image')
        >>> results = {
        ...     'img': img,
        ...     'img_fields': ['img']
        ... }
        >>> # pad to a fixed size
        >>> pad = Pad(size=(320, 416))
        >>> results = pad(results)
        >>> img = imread('/path/of/your/image')
        >>> results = {
        ...     'img': img,
        ...     'img_fields': ['img']
        ... }
        >>> # pad to the minimum size that is divisible by some number
        >>> pad = Pad(size_divisor=32)
        >>> results = pad(results)
    """

    def __init__(self,
                 size: Optional[tuple] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: float = 0,
                 seg_pad_val: float = 255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        assert 'img_fields' in results, \
            '"img_fields" is a required key of results'

        for key in results['img_fields']:
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=self.pad_val)
            results[key] = padded_img
        # 'pad_shape', 'pad_fixed_size' and 'pad_size_divisor' will be used by
        # other pipelines, so they are added to results
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results: dict) -> None:
        """Pad masks according to ``results['pad_shape']``."""
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            results[key] = results[key].pad(pad_shape, pad_val=self.pad_val)

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'seg_pad_val={self.seg_pad_val})'
        return repr_str
