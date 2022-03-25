# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_random_params, cache_randomness

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]

# Indicator for required but missing keys in results
NotInResults = object()

# Import nullcontext if python>=3.7, otherwise use a simple alternative
# implementation.
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(resource=None):
        try:
            yield resource
        finally:
            pass


class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be composed.

    Examples:
        >>> pipeline = [
        >>>     dict(type='Compose',
        >>>         transforms=[
        >>>             dict(type='LoadImageFromFile'),
        >>>             dict(type='Normalize')
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self, transforms: Union[Transform, List[Transform]]):
        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        for t in self.transforms:
            results = t(results)
            if results is None:
                return None
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class KeyMapper(BaseTransform):
    """A transform wrapper to map and reorganize the input/output of the
    wrapped transforms (or sub-pipeline).

    Args:
        transforms (list[dict | callable], optional): Sequence of transform
            object or config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            ``transform`` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        remapping (dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            ``mapping``. Default: None.
        auto_remap (bool, optional): If True, an inverse of the mapping will
            be used as the remapping. If auto_remap is not given, it will be
            automatically set True if 'remapping' is not given, and vice
            versa. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.

    Examples:
        >>> # Example 1: KeyMapper 'gt_img' to 'img'
        >>> pipeline = [
        >>>     # Use KeyMapper to convert outer (original) field name
        >>>     # 'gt_img' to inner (used by inner transforms) filed name
        >>>     # 'img'
        >>>     dict(type='KeyMapper',
        >>>         mapping=dict(img='gt_img'),
        >>>         # auto_remap=True means output key mapping is the revert of
        >>>         # the input key mapping, e.g. inner 'img' will be mapped
        >>>         # back to outer 'gt_img'
        >>>         auto_remap=True,
        >>>         transforms=[
        >>>             # In all transforms' implementation just use 'img'
        >>>             # as a standard field name
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
        >>> # Example 2: Collect and structure multiple items
        >>> pipeline = [
        >>>     # The inner field 'imgs' will be a dict with keys 'img_src'
        >>>     # and 'img_tar', whose values are outer fields 'img1' and
        >>>     # 'img2' respectively.
        >>>     dict(type='KeyMapper',
        >>>         dict(
        >>>             type='KeyMapper',
        >>>             mapping=dict(
        >>>                 imgs=dict(
        >>>                     img_src='img1',
        >>>                     img_tar='img2')),
        >>>         transforms=...)
        >>> ]
    """

    def __init__(self,
                 transforms: Union[Transform, List[Transform]] = None,
                 mapping: Optional[Dict] = None,
                 remapping: Optional[Dict] = None,
                 auto_remap: Optional[bool] = None,
                 allow_nonexist_keys: bool = False):

        self.allow_nonexist_keys = allow_nonexist_keys
        self.mapping = mapping

        if auto_remap is None:
            auto_remap = remapping is None
        self.auto_remap = auto_remap

        if self.auto_remap:
            if remapping is not None:
                raise ValueError('KeyMapper: ``remapping`` must be None if'
                                 '`auto_remap` is set True.')
            self.remapping = mapping
        else:
            self.remapping = remapping

        if transforms is None:
            transforms = []
        self.transforms = Compose(transforms)

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def map_input(self, data: Dict, mapping: Dict) -> Dict[str, Any]:
        """KeyMapper inputs for the wrapped transforms by gathering and
        renaming data items according to the mapping.

        Args:
            data (dict): The original input data
            mapping (dict): The input key mapping. See the document of
                ``mmcv.transforms.wrappers.KeyMapper`` for details.

        Returns:
            dict: The input data with remapped keys. This will be the actual
                input of the wrapped pipeline.
        """

        def _map(data, m):
            if isinstance(m, dict):
                # m is a dict {inner_key:outer_key, ...}
                return {k_in: _map(data, k_out) for k_in, k_out in m.items()}
            if isinstance(m, (tuple, list)):
                # m is a list or tuple [outer_key1, outer_key2, ...]
                # This is the case when we collect items from the original
                # data to form a list or tuple to feed to the wrapped
                # transforms.
                return m.__class__(_map(data, e) for e in m)

            # m is an outer_key
            if self.allow_nonexist_keys:
                return data.get(m, NotInResults)
            else:
                return data.get(m)

        collected = _map(data, mapping)
        collected = {
            k: v
            for k, v in collected.items() if v is not NotInResults
        }

        # Retain unmapped items
        inputs = data.copy()
        inputs.update(collected)

        return inputs

    def map_output(self, data: Dict, remapping: Dict) -> Dict[str, Any]:
        """KeyMapper outputs from the wrapped transforms by gathering and
        renaming data items according to the remapping.

        Args:
            data (dict): The output of the wrapped pipeline.
            remapping (dict): The output key mapping. See the document of
                ``mmcv.transforms.wrappers.KeyMapper`` for details.

        Returns:
            dict: The output with remapped keys.
        """

        def _map(data, m):
            if isinstance(m, dict):
                assert isinstance(data, dict)
                results = {}
                for k_in, k_out in m.items():
                    assert k_in in data
                    results.update(_map(data[k_in], k_out))
                return results
            if isinstance(m, (list, tuple)):
                assert isinstance(data, (list, tuple))
                assert len(data) == len(m)
                results = {}
                for m_i, d_i in zip(m, data):
                    results.update(_map(d_i, m_i))
                return results

            return {m: data}

        # Note that unmapped items are not retained, which is different from
        # the behavior in map_input. This is to avoid original data items
        # being overwritten by intermediate namesakes
        return _map(data, remapping)

    def transform(self, results: Dict) -> Dict:

        inputs = self.map_input(results, self.mapping)
        outputs = self.transforms(inputs)

        if self.remapping:
            outputs = self.map_output(outputs, self.remapping)

        results.update(outputs)
        return results


@TRANSFORMS.register_module()
class TransformBroadcaster(KeyMapper):
    """A transform wrapper to apply the wrapped transforms to multiple data
    items. For example, apply Resize to multiple images.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        mapping (dict): A dict that defines the input key mapping.
            Note that to apply the transforms to multiple data items, the
            outer keys of the target items should be remapped as a list with
            the standard inner key (The key required by the wrapped transform).
            See the following example and the document of
            ``mmcv.transforms.wrappers.KeyMapper`` for details.
        remapping (dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            ``mapping``. Default: None.
        auto_remap (bool, optional): If True, an inverse of the mapping will
            be used as the remapping. If auto_remap is not given, it will be
            automatically set True if 'remapping' is not given, and vice
            versa. Default: None.
        allow_nonexist_keys (bool): If False, the outer keys in the mapping
            must exist in the input data, or an exception will be raised.
            Default: False.
        share_random_params (bool): If True, the random transform
            (e.g., RandomFlip) will be conducted in a deterministic way and
            have the same behavior on all data items. For example, to randomly
            flip either both input image and ground-truth image, or none.
            Default: False.

    .. note::
        To apply the transforms to each elements of a list or tuple, instead
        of separating data items, you can map the outer key of the target
        sequence to the standard inner key. See example 2.
        example.

    Examples:
        >>> # Example 1:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 1: from multiple outer fields
        >>>         mapping=dict(img=['lq', 'gt']),
        >>>         auto_remap=True,
        >>>         # share_random_param=True means using identical random
        >>>         # parameters in every processing
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
        >>> # Example 2:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 2: from one outer field that contains multiple
        >>>         # data elements (e.g. a list)
        >>>         # mapping=dict(img='images'),
        >>>         auto_remap=True,
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable[[Dict], Dict]]],
                 mapping: Optional[Dict] = None,
                 remapping: Optional[Dict] = None,
                 auto_remap: Optional[bool] = None,
                 allow_nonexist_keys: bool = False,
                 share_random_params: bool = False):
        super().__init__(transforms, mapping, remapping, auto_remap,
                         allow_nonexist_keys)

        self.share_random_params = share_random_params

    def scatter_sequence(self, data: Dict) -> List[Dict]:
        # infer split number from input
        seq_len = None
        key_rep = None
        for key in self.mapping:

            assert isinstance(data[key], Sequence)
            if seq_len is not None:
                if len(data[key]) != seq_len:
                    raise ValueError('Got inconsistent sequence length: '
                                     f'{seq_len} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                seq_len = len(data[key])
                key_rep = key

        scatters = []
        for i in range(seq_len):
            scatter = data.copy()
            for key in self.mapping:
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def transform(self, results: Dict):
        # Apply input remapping
        inputs = self.map_input(results, self.mapping)

        # Scatter sequential inputs into a list
        inputs = self.scatter_sequence(inputs)

        # Control random parameter sharing with a context manager
        if self.share_random_params:
            # The context manager :func`:cache_random_params` will let
            # cacheable method of the transforms cache their outputs. Thus
            # the random parameters will only generated once and shared
            # by all data items.
            ctx = cache_random_params
        else:
            ctx = nullcontext

        with ctx(self.transforms):
            outputs = [self.transforms(_input) for _input in inputs]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in outputs]
            for key in outputs[0]
        }

        # Apply output remapping
        if self.remapping:
            outputs = self.map_output(outputs, self.remapping)

        results.update(outputs)
        return results


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen pipeline from given candidates.

    Args:
        transforms (list[list]): A list of pipeline candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Transform, List[Transform]]],
                 prob: Optional[List[float]] = None):

        if prob is not None:
            assert mmcv.is_seq_of(prob, float)
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_pipeline_index(self):
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results):
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)
