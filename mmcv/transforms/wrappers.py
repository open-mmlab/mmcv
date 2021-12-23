# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

import mmcv
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_random_params

# Indicator for required but missing keys in results
NotInResults = object()


@TRANSFORMS.register_module()
class Compose(BaseTransform):
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
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

    def transform(self, results):
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict: Transformed results.
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
        transforms (list[dict|callable]):
        input_mapping (dict): A dict that defines the input key mapping.
            The keys corresponds to the inner key (i.e. kwargs of the
            `transform` method), and the values corresponds to the outer
            keys (i.e. the keys of the data/results).
        output_mapping(dict): A dict that defines the output key mapping.
            The keys corresponds to the inner key (i.e. the keys of the
            output dict of the `transform` method), and the values
            corresponds to the outer keys (i.e. the keys of the
            data/results).
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

        if inplace:
            if not self.strict:
                raise ValueError('Remap: `strict` must be set True if'
                                 '`inplace` is set True.')

            if output_mapping is not None:
                raise ValueError('Remap: the output_mapping must be None '
                                 'if `inplace` is set True.')
            self.output_mapping = input_mapping
        else:
            self.output_mapping = output_mapping

        self.transforms = Compose(transforms)

    def remap_input(self, data: Dict, input_mapping: Dict) -> Dict[str, Any]:
        """Remap inputs for the wrapped transforms by gathering and renaming
        data items according to the input_mapping.

        Args:
            data (dict): The original data dictionary of the pipeline
            input_mapping(dict):
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
        data items according to the output_mapping."""

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

    def scatter_sequence(self, data: Dict) -> list[dict]:
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
