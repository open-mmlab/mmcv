from collections.abc import Sequence

from mmcv.utils import build_from_cfg
from .base_transform import TRANSFORMS, BaseTransform


class DataPipeline:

    def __init__(self, transforms):
        if not isinstance(transforms, Sequence):
            raise TypeError('transforms must be a sequence')
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, TRANSFORMS)
                self.transforms.append(transform)
            elif isinstance(transform, BaseTransform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    'each transform must be a BaseTransform object or a dict')
        self._validate_pipeline()

    def __call__(self, data):
        """Apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict | None: Transformed data. If any transform returns None, then \
               the pipeline also returns None.
        """
        if not isinstance(data, dict):
            raise TypeError(f'data should be a dict, but got {type(data)}')
        if len(self.transforms) > 0:
            if not set(self.transforms[0].required_keys) <= set(data.keys()):
                missing_keys = set(self.transforms[0].required_keys) - set(
                    data.keys())
                raise KeyError(f'Missing keys {missing_keys} in the data dict')
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t},'
        format_string += '\n)'
        return format_string

    def _validate_pipeline(self):
        """Validate if the pipeline is proper.

        Returns:
            bool: True if the pipeline is proper.
        """
        keys = set()
        for i, t in enumerate(self.transforms):
            required_keys = set(t.required_keys)
            if i > 0 and not keys >= set(required_keys):
                raise KeyError(
                    f'transform {t.__class__.__name__} requires '
                    f'{required_keys - keys} but it is not provided after '
                    f'transform {self.transforms[i - 1].__class__.__name__}')
            keys.update(t.updated_keys)
