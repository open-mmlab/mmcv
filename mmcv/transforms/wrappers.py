# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_random_params

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


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms: List[Union[Dict, Callable[[Dict], Dict]]]):
        assert isinstance(transforms, Sequence)
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
class Remap(BaseTransform):
    """A transform wrapper to remap and reorganize the input/output of the
    wrapped transforms (or sub-pipeline).

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        input_mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e., kwargs of the
            `transform` method), and should be string type. The values
            corresponds to the outer keys (i.e., the keys of the
            data/results), and should have a type of string, list or dict.
            None means not applying input mapping. Default: None.
        output_mapping(dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            `input_mapping`. Default: None.
        inplace (bool): If True, an inverse of the input_mapping will be used
            as the output_mapping. Note that if inplace is set True,
            output_mapping should be None and strict should be True.
            Default: False.
        strict (bool): If True, the outer keys in the input_mapping must exist
            in the input data, or an excaption will be raised. If False,
            the missing keys will be assigned a special value `NotInResults`
            during input remapping. Default: True.

    Examples:
        >>> # Example 1: Remap 'gt_img' to 'img'
        >>> pipeline = [
        >>>     # Use Remap to convert outer (original) field name 'gt_img'
        >>>     # to inner (used by inner transforms) filed name 'img'
        >>>     dict(type='Remap',
        >>>         input_mapping=dict(img='gt_img'),
        >>>         # inplace=True means output key mapping is the revert of
        >>>         # the input key mapping, e.g. inner 'img' will be mapped
        >>>         # back to outer 'gt_img'
        >>>         inplace=True,
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
        >>>     dict(type='Remap',
        >>>         dict(
        >>>             type='Remap',
        >>>             input_mapping=dict(
        >>>                 imgs=dict(
        >>>                     img_src='img1',
        >>>                     img_tar='img2')),
        >>>         transforms=...)
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable[[Dict], Dict]]],
                 input_mapping: Optional[Dict] = None,
                 output_mapping: Optional[Dict] = None,
                 inplace: bool = False,
                 strict: bool = True):

        self.inplace = inplace
        self.strict = strict
        self.input_mapping = input_mapping

        if self.inplace:
            if not self.strict:
                raise ValueError('Remap: `strict` must be set True if'
                                 '`inplace` is set True.')

            if output_mapping is not None:
                raise ValueError('Remap: `output_mapping` must be None if'
                                 '`inplace` is set True.')
            self.output_mapping = input_mapping
        else:
            self.output_mapping = output_mapping

        self.transforms = Compose(transforms)

    def remap_input(self, data: Dict, input_mapping: Dict) -> Dict[str, Any]:
        """Remap inputs for the wrapped transforms by gathering and renaming
        data items according to the input_mapping.

        Args:
            data (dict): The original input data
            input_mapping(dict): The input key mapping. See the document of
                mmcv.transforms.wrappers.Remap` for details.
        Returns:
            dict: The input data with remapped keys. This will be the actual
                input of the wrapped pipeline.
        """

        def _remap(data, m):
            if isinstance(m, dict):
                # m is a dict {inner_key:outer_key, ...}
                return {k_in: _remap(data, k_out) for k_in, k_out in m.items()}
            if isinstance(m, (tuple, list)):
                # m is a list [outer_key1, outer_key2, ...]
                return m.__class__(_remap(data, e) for e in m)

            # m is an outer_key
            if self.strict:
                return data.get(m)
            else:
                return data.get(m, NotInResults)

        collected = _remap(data, input_mapping)
        collected = {
            k: v
            for k, v in collected.items() if v is not NotInResults
        }

        # Retain unmapped items
        inputs = data.copy()
        inputs.update(collected)

        return inputs

    def remap_output(self, data: Dict, output_mapping: Dict) -> Dict[str, Any]:
        """Remap outputs from the wrapped transforms by gathering and renaming
        data items according to the output_mapping.

        Args:
            data (dict): The output of the wrapped pipeline.
            input_mapping(dict): The output key mapping. See the document of
                `mmcv.transforms.wrappers.Remap` for details.

        Returns:
            dict: The output with remapped keys.
        """

        def _remap(data, m):
            if isinstance(m, dict):
                assert isinstance(data, dict)
                results = {}
                for k_in, k_out in m.items():
                    assert k_in in data
                    results.update(_remap(data[k_in], k_out))
                return results
            if isinstance(m, (list, tuple)):
                assert isinstance(data, (list, tuple))
                assert len(data) == len(m)
                results = {}
                for m_i, d_i in zip(m, data):
                    results.update(_remap(d_i, m_i))
                return results

            if data == NotInResults:
                raise ValueError(
                    f'Attempt to assign `NotInResults` to output key {m}.'
                    '`NotInResults` just serves as a placeholder for missing '
                    'keys in non-strict input mapping. It should not be '
                    'assigned to any output.')
            return {m: data}

        # Note that unmapped items are not retained, which is different from
        # the behavior in remap_input. This is to avoid original data items
        # being overwritten by intermediate namesakes
        return _remap(data, output_mapping)

    def transform(self, results: Dict) -> Dict:

        inputs = self.remap_input(results, self.input_mapping)
        outputs = self.transforms(inputs)

        if self.output_mapping:
            outputs = self.remap_output(outputs, self.output_mapping)

        results.update(outputs)
        return results


@TRANSFORMS.register_module()
class ApplyToMultiple(Remap):
    """A transform wrapper to apply the wrapped transforms to multiple data
    items. For example, apply Resize to multiple images.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be wrapped.
        input_mapping (dict): A dict that defines the input key mapping.
            Note that to apply the transforms to multiple data items, the
            outer keys of the target items should be remapped as a list with
            the standard inner key (The key required by the wrapped transform).
            See the following example and the document of
            `mmcv.transforms.wrappers.Remap` for details.
        output_mapping(dict): A dict that defines the output key mapping.
            The keys and values have the same meanings and rules as in the
            `input_mapping`. Default: None.
        inplace (bool): If True, an inverse of the input_mapping will be used
            as the output_mapping. Note that if inplace is set True,
            output_mapping should be None and strict should be True.
            Default: False.
        strict (bool): If True, the outer keys in the input_mapping must exist
            in the input data, or an excaption will be raised. If False,
            the missing keys will be assigned a special value `NotInResults`
            during input remapping. Default: True.
        share_random_params (bool): If True, the random transform
            (e.g., RandomFlip) will be conducted in a deterministic way and
            have the same behavior on all data items. For example, to randomly
            flip either both input image and ground-truth image, or none.
            Default: False.

    .. note::
        To apply the transforms to each elements of a list or tuple, instead
        of separate data items, you can remap the outer key of the target
        sequence to the standard inner key. See example 2.
        example.

    Examples:
        >>> # Example 1:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile', key='lq'),  # low-quality img
        >>>     dict(type='LoadImageFromFile', key='gt'),  # ground-truth img
        >>>     # ApplyToMultiple maps multiple outer fields to standard the
        >>>     # inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='ApplyToMultiple',
        >>>         # case 1: from multiple outer fields
        >>>         input_mapping=dict(img=['lq', 'gt']),
        >>>         inplace=True,
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
        >>>     # ApplyToMultiple maps multiple outer fields to standard the
        >>>     # inner field and process them with wrapped transforms
        >>>     # respectively
        >>>     dict(type='ApplyToMultiple',
        >>>         # case 2: from one outer field that contains multiple
        >>>         # data elements (e.g. a list)
        >>>         # input_mapping=dict(img='images'),
        >>>         inplace=True,
        >>>         share_random_param=True,
        >>>         transforms=[
        >>>             dict(type='Crop', crop_size=(384, 384)),
        >>>             dict(type='Normalize'),
        >>>         ])
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Dict, Callable[[Dict], Dict]]],
                 input_mapping: Optional[Dict] = None,
                 output_mapping: Optional[Dict] = None,
                 inplace: bool = False,
                 strict: bool = True,
                 share_random_params: bool = False):
        super().__init__(transforms, input_mapping, output_mapping, inplace,
                         strict)

        self.share_random_params = share_random_params

    def scatter_sequence(self, data: Dict) -> List[Dict]:
        # infer split number from input
        seq_len = 0
        key_rep = None
        for key in self.input_mapping.keys():

            assert isinstance(data[key], Sequence)
            if seq_len:
                if len(data[key]) != seq_len:
                    raise ValueError('Got inconsistent sequence length: '
                                     f'{seq_len} ({key_rep}) vs. '
                                     f'{len(data[key])} ({key})')
            else:
                seq_len = len(data[key])
                key_rep = key

        if not seq_len:
            raise RuntimeError(
                'Fail to infer the sequence length. Please ensure that '
                'the input items are sequences with the same length.')

        scatters = []
        for i in range(seq_len):
            scatter = data.copy()
            for key in self.input_mapping.keys():
                scatter[key] = data[key][i]
            scatters.append(scatter)
        return scatters

    def transform(self, results: Dict):
        # Apply input remapping
        inputs = self.remap_input(results, self.input_mapping)

        # Scatter sequential inputs into a list
        inputs = self.scatter_sequence(inputs)

        # Control random parameter sharing with a contextmanager
        if self.share_random_params:
            cm = cache_random_params
        else:
            cm = nullcontext

        with cm(self.transforms):
            outputs = [self.transforms(_input) for _input in inputs]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in outputs]
            for key in outputs[0].keys()
        }

        # Apply output remapping
        if self.output_mapping:
            outputs = self.remap_output(outputs, self.output_mapping)

        results.update(outputs)
        return results


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen pipeline from given candidates.

    Args:
        pipelines (list[list]): A list of pipeline candidates, each is a
            sequence of transforms.
        pipeline_probs (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         pipelines=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 pipelines: List[List[Union[Dict, Callable[[Dict], Dict]]]],
                 pipeline_probs: Optional[List[float]] = None):

        if pipeline_probs is not None:
            assert mmcv.is_seq_of(pipeline_probs, float)
            assert len(pipelines) == len(pipeline_probs), \
                '`pipelines` and `pipeline_probs` must have same lengths. ' \
                f'Got {len(pipelines)} vs {len(pipeline_probs)}.'
            assert sum(pipeline_probs) == 1

        self.pipeline_probs = pipeline_probs
        self.pipelines = [Compose(transforms) for transforms in pipelines]

    def transform(self, results):
        pipeline = np.random.choice(self.pipelines, p=self.pipeline_probs)
        return pipeline(results)
