from collections.abc import Sequence

import numpy as np

import mmcv
from ..builder import PIPELINES


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

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
        >>> img = imread('/path/of/your/img')
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

        The required key of ``results`` is ``img_fields``, which is a list of
        string. Every item of it is also the key of ``results``, whose value is
        an image.

        Args:
            results (dict): Result dict from loading pipeline.

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
