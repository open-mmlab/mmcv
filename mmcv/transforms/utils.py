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
    """Decorator that mark a method of a transform class as a cacheable method.

    This decorator is usually used together with the context-manager
    cache_random_params. In this context, a cacheable method will cache its
    return value(s) at the first time being invoked, and always return the
    cached values when being invoked again.

    .. note::
        only a instance method can be decorated as a cacheable_method.
    """

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
            # Initialize the cache of the transform instances. The flag
            # `cache_enabled` is set by contextmanagers like
            # `cache_random_params`.
            if not hasattr(instance, '_cache'):
                setattr(instance, '_cache', {})

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, '_cache'):
                del instance._cache
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        if self.instance_ref is None:
            self.instance_ref = weakref.ref(obj)
        return self


@contextmanager
def cache_random_params(transforms: Union[BaseTransform, Iterable]):

    # key2method stores the original methods that are replaced by the wrapped
    # ones. These method will be restituted when exiting the context.
    key2method = dict()

    # key2counter stores the usage number of each cacheable_method. This is
    # used to check that any cacheable_method is invoked once during processing
    # on data sample.
    key2counter = defaultdict(int)

    def _add_counter(obj, method_name):
        method = getattr(obj, method_name)
        key = f'{id(obj)}.{method_name}'
        key2method[key] = method

        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            key2counter[key] += 1
            return method(*args, **kwargs)

        return wrapped

    def _start_cache(t: BaseTransform):
        # Set cache enabled flag
        setattr(t, '_cache_enabled', True)

        # Store the original method and init the counter
        if hasattr(t, '_cacheable_methods'):
            setattr(t, 'transform', _add_counter(t, 'transform'))
            for name in t._cacheable_methods:
                setattr(t, name, _add_counter(t, name))

    def _end_cache(t: BaseTransform):
        # Remove cache enabled flag
        del t._cache_enabled

        # Restore the original method
        if hasattr(t, '_cacheable_methods'):
            key_transform = f'{id(t)}.transform'
            for name in t._cacheable_methods:
                key = f'{id(t)}.{name}'
                if key2counter[key] != key2counter[key_transform]:
                    raise RuntimeError(
                        'The cacheable method should be called once and only'
                        f'once during processing one data sample. {t} got'
                        f'unmatched number of {key2counter[key]} ({name}) vs'
                        f'{key2counter[key_transform]} (data samples)')
                setattr(t, name, key2method[key])
            setattr(t, 'transform', key2method[key_transform])

    def _apply(t: Union[BaseTransform, Iterable],
               func: Callable[[BaseTransform], None]):
        if isinstance(t, BaseTransform):
            if hasattr(t, '_cacheable_methods'):
                func(t)
        else:
            for _t in t:
                _apply(_t, func)

    try:
        _apply(transforms, _start_cache)
        yield
    finally:
        _apply(transforms, _end_cache)
