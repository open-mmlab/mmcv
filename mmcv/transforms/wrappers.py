# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_random_params, cache_randomness

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]

# Indicator of keys marked by KeyMapper._map_input, which means ignoring the
# marked keys in KeyMapper._apply_transform so they will be invisible to
# wrapped transforms.
# This can be 2 possible case:
# 1. The key is required but missing in results
# 2. The key is manually set as ... (Ellipsis) in ``mapping``, which means
# the original value in results should be ignored
IgnoreKey = object()

# Import nullcontext if python>=3.7, otherwise use a simple alternative
# implementation.
try:
    from contextlib import nullcontext  # type: ignore
except ImportError:
    from contextlib import contextmanager

    @contextmanager  # type: ignore
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

    def __init__(self, transforms: Union[Transform, Sequence[Transform]]):
        super().__init__()

        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms: List = []
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
            results = t(results)  # type: ignore
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
        >>>         mapping={'img': 'gt_img'},
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

        >>> # Example 3: Manually set ignored keys by "..."
        >>> pipeline = [
        >>>     ...
        >>>     dict(type='KeyMapper',
        >>>         mapping={
        >>>             # map outer key "gt_img" to inner key "img"
        >>>             'img': 'gt_img',
        >>>             # ignore outer key "mask"
        >>>             'mask': ...,
        >>>         },
        >>>         transforms=[
        >>>             dict(type='RandomFlip'),
        >>>         ])
        >>>     ...
        >>> ]
    """

    def __init__(self,
                 transforms: Union[Transform, List[Transform]] = None,
                 mapping: Optional[Dict] = None,
                 remapping: Optional[Dict] = None,
                 auto_remap: Optional[bool] = None,
                 allow_nonexist_keys: bool = False):

        super().__init__()

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

    def _map_input(self, data: Dict,
                   mapping: Optional[Dict]) -> Dict[str, Any]:
        """KeyMapper inputs for the wrapped transforms by gathering and
        renaming data items according to the mapping.

        Args:
            data (dict): The original input data
            mapping (dict, optional): The input key mapping. See the document
                of ``mmcv.transforms.wrappers.KeyMapper`` for details. In
                set None, return the input data directly.

        Returns:
            dict: The input data with remapped keys. This will be the actual
                input of the wrapped pipeline.
        """

        if mapping is None:
            return data.copy()

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

            # allow manually mark a key to be ignored by ...
            if m is ...:
                return IgnoreKey

            # m is an outer_key
            if self.allow_nonexist_keys:
                return data.get(m, IgnoreKey)
            else:
                return data.get(m)

        collected = _map(data, mapping)

        # Retain unmapped items
        inputs = data.copy()
        inputs.update(collected)

        return inputs

    def _map_output(self, data: Dict,
                    remapping: Optional[Dict]) -> Dict[str, Any]:
        """KeyMapper outputs from the wrapped transforms by gathering and
        renaming data items according to the remapping.

        Args:
            data (dict): The output of the wrapped pipeline.
            remapping (dict, optional): The output key mapping. See the
                document of ``mmcv.transforms.wrappers.KeyMapper`` for
                details. If ``remapping is None``, no key mapping will be
                applied but only remove the special token ``IgnoreKey``.

        Returns:
            dict: The output with remapped keys.
        """

        # Remove ``IgnoreKey``
        if remapping is None:
            return {k: v for k, v in data.items() if v is not IgnoreKey}

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

            # ``m is ...`` means the key is marked ignored, in which case the
            # inner resuls will not affect the outer results in remapping.
            # Another case that will have ``data is IgnoreKey`` is that the
            # key is missing in the inputs. In this case, if the inner key is
            # created by the wrapped transforms, it will be remapped to the
            # corresponding outer key during remapping.
            if m is ... or data is IgnoreKey:
                return {}

            return {m: data}

        # Note that unmapped items are not retained, which is different from
        # the behavior in _map_input. This is to avoid original data items
        # being overwritten by intermediate namesakes
        return _map(data, remapping)

    def _apply_transforms(self, inputs: Dict) -> Dict:
        """Apply ``self.transforms``.

        Note that the special token ``IgnoreKey`` will be invisible to
        ``self.transforms``, but not removed in this method. It will be
        eventually removed in :func:``self._map_output``.
        """
        results = inputs.copy()
        inputs = {k: v for k, v in inputs.items() if v is not IgnoreKey}
        outputs = self.transforms(inputs)

        if outputs is None:
            raise ValueError(
                f'Transforms wrapped by {self.__class__.__name__} should '
                'not return None.')

        results.update(outputs)  # type: ignore
        return results

    def transform(self, results: Dict) -> Dict:
        """Apply mapping, wrapped transforms and remapping."""

        # Apply mapping
        inputs = self._map_input(results, self.mapping)
        # Apply wrapped transforms
        outputs = self._apply_transforms(inputs)
        # Apply remapping
        outputs = self._map_output(outputs, self.remapping)

        results.update(outputs)  # type: ignore
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f', mapping = {self.mapping}'
        repr_str += f', remapping = {self.remapping}'
        repr_str += f', auto_remap = {self.auto_remap}'
        repr_str += f', allow_nonexist_keys = {self.allow_nonexist_keys})'
        return repr_str


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
        >>> # Example 1: Broadcast to enumerated keys, each contains a single
        >>> # data element
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 1: from multiple outer fields
        >>>         mapping={'img': ['lq', 'gt']},
        >>>         auto_remap=True,
        >>>         # share_random_param=True means using identical random
        >>>         # parameters in every processing
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]

        >>> # Example 2: Broadcast to keys that contains data sequences
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # TransformBroadcaster maps multiple outer fields to standard
        >>>     # the inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='TransformBroadcaster',
        >>>         # case 2: from one outer field that contains multiple
        >>>         # data elements (e.g. a list)
        >>>         # mapping={'img': 'images'},
        >>>         auto_remap=True,
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]

        >>> Example 3: Set ignored keys in broadcasting
        >>> pipeline = [
        >>>        dict(type='TransformBroadcaster',
        >>>            # Broadcast the wrapped transforms to multiple images
        >>>            # 'lq' and 'gt, but only update 'img_shape' once
        >>>            mapping={
        >>>                'img': ['lq', 'gt'],
        >>>                'img_shape': ['img_shape', ...],
        >>>             },
        >>>            auto_remap=True,
        >>>            share_random_params=True,
        >>>            transforms=[
        >>>                # `RandomCrop` will modify the field "img",
        >>>                # and optionally update "img_shape" if it exists
        >>>                dict(type='RandomCrop'),
        >>>            ])
        >>>    ]
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
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms."""

        # infer split number from input
        seq_len = 0
        key_rep = None

        if self.mapping:
            keys = self.mapping.keys()
        else:
            keys = data.keys()

        for key in keys:
            assert isinstance(data[key], Sequence)
            if seq_len:
                if len(data[key]) != seq_len:
                    raise ValueError('Got inconsistent sequence length: '
                                     f'{seq_len} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                seq_len = len(data[key])
                key_rep = key

        assert seq_len > 0, 'Fail to get the number of broadcasting targets'

        scatters = []
        for i in range(seq_len):  # type: ignore
            scatter = data.copy()
            for key in keys:
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def transform(self, results: Dict):
        """Broadcast wrapped transforms to multiple targets."""

        # Apply input remapping
        inputs = self._map_input(results, self.mapping)

        # Scatter sequential inputs into a list
        input_scatters = self.scatter_sequence(inputs)

        # Control random parameter sharing with a context manager
        if self.share_random_params:
            # The context manager :func`:cache_random_params` will let
            # cacheable method of the transforms cache their outputs. Thus
            # the random parameters will only generated once and shared
            # by all data items.
            ctx = cache_random_params  # type: ignore
        else:
            ctx = nullcontext  # type: ignore

        with ctx(self.transforms):
            output_scatters = [
                self._apply_transforms(_input) for _input in input_scatters
            ]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in output_scatters]
            for key in output_scatters[0]
        }

        # Apply remapping
        outputs = self._map_output(outputs, self.remapping)

        results.update(outputs)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f', mapping = {self.mapping}'
        repr_str += f', remapping = {self.remapping}'
        repr_str += f', auto_remap = {self.auto_remap}'
        repr_str += f', allow_nonexist_keys = {self.allow_nonexist_keys}'
        repr_str += f', share_random_params = {self.share_random_params})'
        return repr_str


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
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

        super().__init__()

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
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class RandomApply(BaseTransform):
    """Apply transforms randomly with a given probability.

    Args:
        transforms (list[dict | callable]): The transform or transform list
            to randomly apply.
        prob (float): The probability to apply transforms. Default: 0.5

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomApply',
        >>>         transforms=[dict(type='HorizontalFlip')],
        >>>         prob=0.3)
        >>> ]
    """

    def __init__(self,
                 transforms: Union[Transform, List[Transform]],
                 prob: float = 0.5):

        super().__init__()
        self.prob = prob
        self.transforms = Compose(transforms)

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_apply(self) -> bool:
        """Return a random bool value indicating whether apply the
        transform."""
        return np.random.rand() < self.prob

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly apply the transform."""
        if self.random_apply():
            return self.transforms(results)  # type: ignore
        else:
            return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f', prob = {self.prob})'
        return repr_str
