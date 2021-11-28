# Copyright (c) OpenMMLab. All rights reserved.

import functools
import inspect
import weakref
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Union

from .base import BaseTransform


class cacheable_method:

    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cacheable_method should only be used to decorate '
                'instance methods (the first argument is `self`).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_cacheable_methods'):
            setattr(owner, '_cacheable_methods', [])
        owner._cacheable_methods.append(self.__name__)

    def __call__(self, *args, **kwargs):
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag `self._cache_enabled`, which should be
        # set by the contextmanagers like `cache_random_parameters`
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Initialize cache and usage counter for cacheable methods
            # of the transform. Specifically, `_cache` is a dict to store the
            # output of cacheable methods; `_cache_usage_counter` is a counter
            # of the usage time of each cached result.
            if not hasattr(instance, '_cache'):
                setattr(instance, '_cache', {})
            if not hasattr(instance, '_cache_usage_counter'):
                setattr(instance, '_cache_usage_counter', defaultdict(int))

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            instance._cache_usage_counter[name] += 1
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, '_cache'):
                del instance._cache
            if hasattr(instance, '_cache_usage_counter'):
                del instance._cache_usage_counter
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        if self.instance_ref is None:
            self.instance_ref = weakref.ref(obj)
        return self


@contextmanager
def cache_random_params(transforms: Union[BaseTransform, Iterable]):

    def _cache_start(t: BaseTransform):
        setattr(t, '_cache_enabled', True)

    def _cache_end(t: BaseTransform):
        del t._cache_enabled

    def _apply(t: Union[BaseTransform, Iterable],
               func: Callable[[BaseTransform], None]):
        if isinstance(t, BaseTransform):
            if hasattr(t, '_cacheable_methods'):
                func(t)
        else:
            for _t in t:
                _apply(_t, func)

    class _RepetitiveCacheChecker():
        """Checker to check that each cacheable method has been called exactly
        once during processing one data sample."""

        def __init__(self, transforms):
            self.transforms = weakref.ref(transforms)

        def _cached_only_once(self, transform):
            if hasattr(transform, '_cacheable_methods'):
                if not hasattr(transform, '_cache_usage_counter'):
                    raise ValueError(
                        f'Cache is not enabled for {transform.__class__}')
                for name in transform._cacheable_methods:
                    if name not in transform._cache_usage_counter:
                        raise ValueError(
                            f'The method {name} of {transform.__class__} has '
                            'not been cached yet!')
                    count = transform._cache_usage_counter[name]
                    if count != 1:
                        raise ValueError(
                            f'The method {name} of {transform.__class__} has '
                            'been called more than once!')

        def check(self):
            _apply(self.transforms, self._cached_only_once)

    try:
        _apply(transforms, _cache_start)
        yield _RepetitiveCacheChecker(transforms)
    finally:
        _apply(transforms, _cache_end)
