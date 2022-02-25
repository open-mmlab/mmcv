# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import pytest

from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.utils import cache_random_params, cacheable_method
from mmcv.transforms.wrappers import (ApplyToMultiple, Compose, RandomChoice,
                                      Remap)


@TRANSFORMS.register_module()
class AddToValue(BaseTransform):
    """Dummy transform to test transform wrappers."""

    def __init__(self, constant_addend=0, use_random_addend=False) -> None:
        super().__init__()
        self.constant_addend = constant_addend
        self.use_random_addend = use_random_addend

    @cacheable_method
    def get_random_addend(self):
        return np.random.rand()

    def transform(self, results):
        augend = results['value']

        if isinstance(augend, list):
            warnings.warn('value is a list', UserWarning)
        if isinstance(augend, dict):
            warnings.warn('value is a dict', UserWarning)

        def _add_to_value(augend, addend):
            if isinstance(augend, list):
                return [_add_to_value(v, addend) for v in augend]
            if isinstance(augend, dict):
                return {k: _add_to_value(v, addend) for k, v in augend.items()}
            return augend + addend

        if self.use_random_addend:
            addend = self.get_random_addend()
        else:
            addend = self.constant_addend

        results['value'] = _add_to_value(results['value'], addend)
        return results


@TRANSFORMS.register_module()
class SumTwoValues(BaseTransform):
    """Dummy transform to test transform wrappers."""

    def transform(self, results):
        if 'num_1' in results and 'num_2' in results:
            results['sum'] = results['num_1'] + results['num_2']
        else:
            results['sum'] = np.nan
        return results


def test_compose():

    # Case 1: build from cfg
    pipeline = [dict(type='AddToValue')]
    pipeline = Compose(pipeline)
    _ = str(pipeline)

    # Case 2: build from transform list
    pipeline = [AddToValue()]
    pipeline = Compose(pipeline)

    # Case 3: invalid build arguments
    pipeline = [[dict(type='AddToValue')]]
    with pytest.raises(TypeError):
        pipeline = Compose(pipeline)

    # Case 4: contain transform with None output
    class DummyTransform(BaseTransform):

        def transform(self, results):
            return None

    pipeline = Compose([DummyTransform()])
    results = pipeline({})
    assert results is None


def test_cache_random_parameters():

    transform = AddToValue(use_random_addend=True)

    # Case 1: cache random parameters
    assert hasattr(AddToValue, '_cacheable_methods')
    assert 'get_random_addend' in AddToValue._cacheable_methods

    with cache_random_params(transform):
        results_1 = transform(dict(value=0))
        results_2 = transform(dict(value=0))
        np.testing.assert_equal(results_1['value'], results_2['value'])

    # Case 2: do not cache random parameters
    results_1 = transform(dict(value=0))
    results_2 = transform(dict(value=0))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(results_1['value'], results_2['value'])

    # Case 3: invalid use of cacheable methods
    with pytest.raises(RuntimeError):
        with cache_random_params(transform):
            _ = transform.get_random_addend()

    # Case 4: apply on nested transforms
    transform = Compose([AddToValue(use_random_addend=True)])
    with cache_random_params(transform):
        results_1 = transform(dict(value=0))
        results_2 = transform(dict(value=0))
        np.testing.assert_equal(results_1['value'], results_2['value'])


def test_remap():

    # Case 1: simple remap
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=1)],
        input_mapping=dict(value='v_in'),
        output_mapping=dict(value='v_out'))

    results = dict(value=0, v_in=1)
    results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in'], 1)
    np.testing.assert_equal(results['v_out'], 2)

    # Case 2: collecting list
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=2)],
        input_mapping=dict(value=['v_in_1', 'v_in_2']),
        output_mapping=dict(value=['v_out_1', 'v_out_2']))
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in_1'], 1)
    np.testing.assert_equal(results['v_in_2'], 2)
    np.testing.assert_equal(results['v_out_1'], 3)
    np.testing.assert_equal(results['v_out_2'], 4)

    # Case 3: collecting dict
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=2)],
        input_mapping=dict(value=dict(v1='v_in_1', v2='v_in_2')),
        output_mapping=dict(value=dict(v1='v_out_1', v2='v_out_2')))
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a dict'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in_1'], 1)
    np.testing.assert_equal(results['v_in_2'], 2)
    np.testing.assert_equal(results['v_out_1'], 3)
    np.testing.assert_equal(results['v_out_2'], 4)

    # Case 4: collecting list with inplace mode
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=2)],
        input_mapping=dict(value=['v_in_1', 'v_in_2']),
        inplace=True)
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v_in_1'], 3)
    np.testing.assert_equal(results['v_in_2'], 4)

    # Case 5: collecting dict with inplace mode
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=2)],
        input_mapping=dict(value=dict(v1='v_in_1', v2='v_in_2')),
        inplace=True)
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a dict'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v_in_1'], 3)
    np.testing.assert_equal(results['v_in_2'], 4)

    # Case 6: nested collection with inplace mode
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=2)],
        input_mapping=dict(value=['v1', dict(v2=['v21', 'v22'], v3='v3')]),
        inplace=True)
    results = dict(value=0, v1=1, v21=2, v22=3, v3=4)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v1'], 3)
    np.testing.assert_equal(results['v21'], 4)
    np.testing.assert_equal(results['v22'], 5)
    np.testing.assert_equal(results['v3'], 6)

    # Case 7: `strict` must be True if `inplace` is set True
    with pytest.raises(ValueError):
        pipeline = Remap(
            transforms=[AddToValue(constant_addend=2)],
            input_mapping=dict(value=['v_in_1', 'v_in_2']),
            inplace=True,
            strict=False)

    # Case 8: output_map must be None if `inplace` is set True
    with pytest.raises(ValueError):
        pipeline = Remap(
            transforms=[AddToValue(constant_addend=1)],
            input_mapping=dict(value='v_in'),
            output_mapping=dict(value='v_out'),
            inplace=True)

    # Case 9: non-strict input mapping
    pipeline = Remap(
        transforms=[SumTwoValues()],
        input_mapping=dict(num_1='a', num_2='b'),
        strict=False)

    results = pipeline(dict(a=1, b=2))
    np.testing.assert_equal(results['sum'], 3)

    results = pipeline(dict(a=1))
    assert np.isnan(results['sum'])

    # Test basic functions
    pipeline = Remap(
        transforms=[AddToValue(constant_addend=1)],
        input_mapping=dict(value='v_in'),
        output_mapping=dict(value='v_out'))

    # __iter__
    for _ in pipeline:
        pass

    # __repr__
    _ = str(pipeline)


def test_apply_to_multiple():

    # Case 1: apply to list in results
    pipeline = ApplyToMultiple(
        transforms=[AddToValue(constant_addend=1)],
        input_mapping=dict(value='values'),
        inplace=True)
    results = dict(values=[1, 2])

    results = pipeline(results)

    np.testing.assert_equal(results['values'], [2, 3])

    # Case 2: apply to multiple keys
    pipeline = ApplyToMultiple(
        transforms=[AddToValue(constant_addend=1)],
        input_mapping=dict(value=['v_1', 'v_2']),
        inplace=True)
    results = dict(v_1=1, v_2=2)

    results = pipeline(results)

    np.testing.assert_equal(results['v_1'], 2)
    np.testing.assert_equal(results['v_2'], 3)

    # Case 3: apply to multiple groups of keys
    pipeline = ApplyToMultiple(
        transforms=[SumTwoValues()],
        input_mapping=dict(num_1=['a_1', 'b_1'], num_2=['a_2', 'b_2']),
        output_mapping=dict(sum=['a', 'b']))

    results = dict(a_1=1, a_2=2, b_1=3, b_2=4)
    results = pipeline(results)

    np.testing.assert_equal(results['a'], 3)
    np.testing.assert_equal(results['b'], 7)

    # Case 4: inconsistent sequence length
    with pytest.raises(ValueError):
        pipeline = ApplyToMultiple(
            transforms=[SumTwoValues()],
            input_mapping=dict(num_1='list_1', num_2='list_2'))

        results = dict(list_1=[1, 2], list_2=[1, 2, 3])
        _ = pipeline(results)

    # Case 5: share random parameter
    pipeline = ApplyToMultiple(
        transforms=[AddToValue(use_random_addend=True)],
        input_mapping=dict(value='values'),
        inplace=True,
        share_random_params=True,
    )

    results = dict(values=[0, 0])
    results = pipeline(results)

    np.testing.assert_equal(results['values'][0], results['values'][1])

    # Test repr
    _ = str(pipeline)


def test_randomchoice():

    # Case 1: given probability
    pipeline = RandomChoice(
        pipelines=[[AddToValue(constant_addend=1.0)],
                   [AddToValue(constant_addend=2.0)]],
        pipeline_probs=[1.0, 0.0])

    results = pipeline(dict(value=1))
    np.testing.assert_equal(results['value'], 2.0)

    # Case 1: default probability
    pipeline = RandomChoice(pipelines=[[AddToValue(
        constant_addend=1.0)], [AddToValue(constant_addend=2.0)]])

    _ = pipeline(dict(value=1))


def test_utils():
    # Test cacheable_method: normal case
    class DummyTransform(BaseTransform):

        @cacheable_method
        def func(self):
            return np.random.rand()

        def transform(self, results):
            _ = self.func()
            return results

    transform = DummyTransform()
    _ = transform({})
    with cache_random_params(transform):
        _ = transform({})

    # Test cacheable_method: invalid function type
    with pytest.raises(TypeError):

        class DummyTransform():

            @cacheable_method
            @staticmethod
            def func():
                return np.random.rand()

    # Test cacheable_method: invalid function argument list
    with pytest.raises(TypeError):

        class DummyTransform():

            @cacheable_method
            def func(cls):
                return np.random.rand()
