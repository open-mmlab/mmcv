# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import pytest

from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.utils import (avoid_cache_randomness, cache_random_params,
                                   cache_randomness)
from mmcv.transforms.wrappers import (Compose, KeyMapper,  RandomApply, RandomChoice,
                                      TransformBroadcaster)


@TRANSFORMS.register_module()
class AddToValue(BaseTransform):
    """Dummy transform to add a given addend to results['value']"""

    def __init__(self, addend=0) -> None:
        super().__init__()
        self.addend = addend

    def add(self, results, addend):
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

        results['value'] = _add_to_value(results['value'], addend)
        return results

    def transform(self, results):
        return self.add(results, self.addend)


@TRANSFORMS.register_module()
class RandomAddToValue(AddToValue):
    """Dummy transform to add a random addend to results['value']"""

    def __init__(self, repeat=1) -> None:
        super().__init__(addend=None)
        self.repeat = repeat

    @cache_randomness
    def get_random_addend(self):
        return np.random.rand()

    def transform(self, results):
        for _ in range(self.repeat):
            results = self.add(results, addend=self.get_random_addend())
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

    transform = RandomAddToValue()

    # Case 1: cache random parameters
    assert hasattr(RandomAddToValue, '_methods_with_randomness')
    assert 'get_random_addend' in RandomAddToValue._methods_with_randomness

    with cache_random_params(transform):
        results_1 = transform(dict(value=0))
        results_2 = transform(dict(value=0))
        np.testing.assert_equal(results_1['value'], results_2['value'])

    # Case 2: do not cache random parameters
    results_1 = transform(dict(value=0))
    results_2 = transform(dict(value=0))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(results_1['value'], results_2['value'])

    # Case 3: allow to invoke random method 0 times
    transform = RandomAddToValue(repeat=0)
    with cache_random_params(transform):
        _ = transform(dict(value=0))

    # Case 4: NOT allow to invoke random method >1 times
    transform = RandomAddToValue(repeat=2)
    with pytest.raises(RuntimeError):
        with cache_random_params(transform):
            _ = transform(dict(value=0))

    # Case 5: apply on nested transforms
    transform = Compose([RandomAddToValue()])
    with cache_random_params(transform):
        results_1 = transform(dict(value=0))
        results_2 = transform(dict(value=0))
        np.testing.assert_equal(results_1['value'], results_2['value'])


def test_key_mapper():

    # Case 1: simple remap
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=1)],
        mapping={'value': 'v_in'},
        remapping={'value': 'v_out'})

    results = dict(value=0, v_in=1)
    results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in'], 1)
    np.testing.assert_equal(results['v_out'], 2)

    # Case 2: collecting list
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=2)],
        mapping={'value': ['v_in_1', 'v_in_2']},
        remapping={'value': ['v_out_1', 'v_out_2']})
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in_1'], 1)
    np.testing.assert_equal(results['v_in_2'], 2)
    np.testing.assert_equal(results['v_out_1'], 3)
    np.testing.assert_equal(results['v_out_2'], 4)

    # Case 3: collecting dict
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=2)],
        mapping={'value': {
            'v1': 'v_in_1',
            'v2': 'v_in_2'
        }},
        remapping={'value': {
            'v1': 'v_out_1',
            'v2': 'v_out_2'
        }})
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a dict'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)  # should be unchanged
    np.testing.assert_equal(results['v_in_1'], 1)
    np.testing.assert_equal(results['v_in_2'], 2)
    np.testing.assert_equal(results['v_out_1'], 3)
    np.testing.assert_equal(results['v_out_2'], 4)

    # Case 4: collecting list with auto_remap mode
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=2)],
        mapping=dict(value=['v_in_1', 'v_in_2']),
        auto_remap=True)
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v_in_1'], 3)
    np.testing.assert_equal(results['v_in_2'], 4)

    # Case 5: collecting dict with auto_remap mode
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=2)],
        mapping=dict(value=dict(v1='v_in_1', v2='v_in_2')),
        auto_remap=True)
    results = dict(value=0, v_in_1=1, v_in_2=2)

    with pytest.warns(UserWarning, match='value is a dict'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v_in_1'], 3)
    np.testing.assert_equal(results['v_in_2'], 4)

    # Case 6: nested collection with auto_remap mode
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=2)],
        mapping=dict(value=['v1', dict(v2=['v21', 'v22'], v3='v3')]),
        auto_remap=True)
    results = dict(value=0, v1=1, v21=2, v22=3, v3=4)

    with pytest.warns(UserWarning, match='value is a list'):
        results = pipeline(results)

    np.testing.assert_equal(results['value'], 0)
    np.testing.assert_equal(results['v1'], 3)
    np.testing.assert_equal(results['v21'], 4)
    np.testing.assert_equal(results['v22'], 5)
    np.testing.assert_equal(results['v3'], 6)

    # Case 7: output_map must be None if `auto_remap` is set True
    with pytest.raises(ValueError):
        pipeline = KeyMapper(
            transforms=[AddToValue(addend=1)],
            mapping=dict(value='v_in'),
            remapping=dict(value='v_out'),
            auto_remap=True)

    # Case 8: allow_nonexist_keys8
    pipeline = KeyMapper(
        transforms=[SumTwoValues()],
        mapping=dict(num_1='a', num_2='b'),
        auto_remap=False,
        allow_nonexist_keys=True)

    results = pipeline(dict(a=1, b=2))
    np.testing.assert_equal(results['sum'], 3)

    results = pipeline(dict(a=1))
    assert np.isnan(results['sum'])

    # Case 9: use wrapper as a transform
    transform = KeyMapper(mapping=dict(b='a'), auto_remap=False)
    results = transform(dict(a=1))
    # note that the original key 'a' will not be removed
    assert results == dict(a=1, b=1)

    # Test basic functions
    pipeline = KeyMapper(
        transforms=[AddToValue(addend=1)],
        mapping=dict(value='v_in'),
        remapping=dict(value='v_out'))

    # __iter__
    for _ in pipeline:
        pass

    # __repr__
    _ = str(pipeline)


def test_apply_to_multiple():

    # Case 1: apply to list in results
    pipeline = TransformBroadcaster(
        transforms=[AddToValue(addend=1)],
        mapping=dict(value='values'),
        auto_remap=True)
    results = dict(values=[1, 2])

    results = pipeline(results)

    np.testing.assert_equal(results['values'], [2, 3])

    # Case 2: apply to multiple keys
    pipeline = TransformBroadcaster(
        transforms=[AddToValue(addend=1)],
        mapping=dict(value=['v_1', 'v_2']),
        auto_remap=True)
    results = dict(v_1=1, v_2=2)

    results = pipeline(results)

    np.testing.assert_equal(results['v_1'], 2)
    np.testing.assert_equal(results['v_2'], 3)

    # Case 3: apply to multiple groups of keys
    pipeline = TransformBroadcaster(
        transforms=[SumTwoValues()],
        mapping=dict(num_1=['a_1', 'b_1'], num_2=['a_2', 'b_2']),
        remapping=dict(sum=['a', 'b']),
        auto_remap=False)

    results = dict(a_1=1, a_2=2, b_1=3, b_2=4)
    results = pipeline(results)

    np.testing.assert_equal(results['a'], 3)
    np.testing.assert_equal(results['b'], 7)

    # Case 4: inconsistent sequence length
    with pytest.raises(ValueError):
        pipeline = TransformBroadcaster(
            transforms=[SumTwoValues()],
            mapping=dict(num_1='list_1', num_2='list_2'),
            auto_remap=False)

        results = dict(list_1=[1, 2], list_2=[1, 2, 3])
        _ = pipeline(results)

    # Case 5: share random parameter
    pipeline = TransformBroadcaster(
        transforms=[RandomAddToValue()],
        mapping=dict(value='values'),
        auto_remap=True,
        share_random_params=True)

    results = dict(values=[0, 0])
    results = pipeline(results)

    np.testing.assert_equal(results['values'][0], results['values'][1])

    # Test repr
    _ = str(pipeline)


def test_random_choice():

    # Case 1: given probability
    pipeline = RandomChoice(
        transforms=[[AddToValue(addend=1.0)], [AddToValue(addend=2.0)]],
        prob=[1.0, 0.0])

    results = pipeline(dict(value=1))
    np.testing.assert_equal(results['value'], 2.0)

    # Case 2: default probability
    pipeline = RandomChoice(transforms=[[AddToValue(
        addend=1.0)], [AddToValue(addend=2.0)]])

    _ = pipeline(dict(value=1))

    # Case 3: nested RandomChoice in TransformBroadcaster
    pipeline = TransformBroadcaster(
        transforms=[
            RandomChoice(
                transforms=[[AddToValue(addend=1.0)],
                            [AddToValue(addend=2.0)]], ),
        ],
        mapping={'value': 'values'},
        auto_remap=True,
        share_random_params=True)

    results = dict(values=[0 for _ in range(10)])
    results = pipeline(results)
    # check share_random_params=True works so that all values are same
    values = results['values']
    assert all(map(lambda x: x == values[0], values))


def test_random_apply():

    # Case 1: simple use
    pipeline = RandomApply(transforms=[AddToValue(addend=1.0)], prob=1.0)
    results = pipeline(dict(value=1))
    np.testing.assert_equal(results['value'], 2.0)

    pipeline = RandomApply(transforms=[AddToValue(addend=1.0)], prob=0.0)
    results = pipeline(dict(value=1))
    np.testing.assert_equal(results['value'], 1.0)

    # Case 2: nested RandomApply in TransformBroadcaster
    pipeline = TransformBroadcaster(
        transforms=[RandomApply(transforms=[AddToValue(addend=1)], prob=0.5)],
        mapping={'value': 'values'},
        auto_remap=True,
        share_random_params=True)

    results = dict(values=[0 for _ in range(10)])
    results = pipeline(results)
    # check share_random_params=True works so that all values are same
    values = results['values']
    assert all(map(lambda x: x == values[0], values))

    # __iter__
    for _ in pipeline:
        pass


def test_utils():
    # Test cache_randomness: normal case
    class DummyTransform(BaseTransform):

        @cache_randomness
        def func(self):
            return np.random.rand()

        def transform(self, results):
            _ = self.func()
            return results

    transform = DummyTransform()
    _ = transform({})
    with cache_random_params(transform):
        _ = transform({})

    # Test cache_randomness: invalid function type
    with pytest.raises(TypeError):

        class DummyTransform(BaseTransform):

            @cache_randomness
            @staticmethod
            def func():
                return np.random.rand()

            def transform(self, results):
                return results

    # Test cache_randomness: invalid function argument list
    with pytest.raises(TypeError):

        class DummyTransform(BaseTransform):

            @cache_randomness
            def func(cls):
                return np.random.rand()

            def transform(self, results):
                return results

    # Test avoid_cache_randomness: invalid mixture with cache_randomness
    with pytest.raises(RuntimeError):

        @avoid_cache_randomness
        class DummyTransform(BaseTransform):

            @cache_randomness
            def func(self):
                pass

            def transform(self, results):
                return results

    # Test avoid_cache_randomness: raise error in cache_random_params
    with pytest.raises(RuntimeError):

        @avoid_cache_randomness
        class DummyTransform(BaseTransform):

            def transform(self, results):
                return results

        transform = DummyTransform()
        with cache_random_params(transform):
            pass

    # Test avoid_cache_randomness: non-inheritable
    @avoid_cache_randomness
    class DummyBaseTransform(BaseTransform):

        def transform(self, results):
            return results

    class DummyTransform(DummyBaseTransform):
        pass

    transform = DummyTransform()
    with cache_random_params(transform):
        pass
