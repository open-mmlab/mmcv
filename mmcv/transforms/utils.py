# Copyright (c) OpenMMLab. All rights reserved.

import functools
import inspect
import weakref
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Callable, Union

from .base import BaseTransform


class cache_randomness:
    """Decorator that marks the method with random return value(s) in a
    transform class.

    This decorator is usually used together with the context-manager
    :func`:cache_random_params`. In this context, a decorated method will
    cache its return value(s) at the first time of being invoked, and always
    return the cached values when being invoked again.

    .. note::
        Only an instance method can be decorated by ``cache_randomness``.
    """

    def __init__(self, func):

        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            raise TypeError('Unsupport callable to decorate with'
                            '@cache_randomness.')
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != 'self':
            raise TypeError(
                '@cache_randomness should only be used to decorate '
                'instance methods (the first argument is ``self``).')

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, '_methods_with_randomness'):
            setattr(owner, '_methods_with_randomness', [])
        owner._methods_with_randomness.append(self.__name__)

    def __call__(self, *args, **kwargs):
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, '_cache_enabled', False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
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
        self.instance_ref = weakref.ref(obj)
        return self


@contextmanager
def cache_random_params(transforms: Union[BaseTransform, Iterable]):
    """Context-manager that enables the cache of return values of methods
    decorated by ``cache_randomness`` in transforms.

    In this mode, decorated methods will cache their return values on the
    first invoking, and always return the cached value afterward. This allow
    to apply random transforms in a deterministic way. For example, apply same
    transforms on multiple examples. See ``cache_randomness`` for more
    information.

    Args:
        transforms (BaseTransform|list[BaseTransform]): The transforms to
            enable cache.
    """

    # key2method stores the original methods that are replaced by the wrapped
    # ones. These methods will be restituted when exiting the context.
    key2method = dict()

    # key2counter stores the usage number of each cache_randomness. This is
    # used to check that any cache_randomness is invoked once during processing
    # on data sample.
    key2counter = defaultdict(int)

    def _add_invoke_counter(obj, method_name):
        method = getattr(obj, method_name)
        key = f'{id(obj)}.{method_name}'
        key2method[key] = method

        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            key2counter[key] += 1
            return method(*args, **kwargs)

        return wrapped

    def _add_invoke_checker(obj, method_name):
        # check that the method in _methods_with_randomness has been
        # invoked at most once
        method = getattr(obj, method_name)
        key = f'{id(obj)}.{method_name}'
        key2method[key] = method

        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            # clear counter
            for name in obj._methods_with_randomness:
                key = f'{id(obj)}.{name}'
                key2counter[key] = 0

            output = method(*args, **kwargs)

            for name in obj._methods_with_randomness:
                key = f'{id(obj)}.{name}'
                if key2counter[key] > 1:
                    raise RuntimeError(
                        'The method decorated by ``cache_randomness`` should '
                        'be invoked at most once during processing one data '
                        f'sample. The method {name} of {obj} has been invoked'
                        f' {key2counter[key]} times.')
            return output

        return wrapped

    def _start_cache(t: BaseTransform):
        # Set cache enabled flag
        setattr(t, '_cache_enabled', True)

        # Store the original method and init the counter
        if hasattr(t, '_methods_with_randomness'):
            setattr(t, 'transform', _add_invoke_checker(t, 'transform'))
            for name in t._methods_with_randomness:
                setattr(t, name, _add_invoke_counter(t, name))

    def _end_cache(t: BaseTransform):
        # Remove cache enabled flag
        del t._cache_enabled
        if hasattr(t, '_cache'):
            del t._cache

        # Restore the original method
        if hasattr(t, '_methods_with_randomness'):
            for name in t._methods_with_randomness:
                key = f'{id(t)}.{name}'
                setattr(t, name, key2method[key])

            key_transform = f'{id(t)}.transform'
            setattr(t, 'transform', key2method[key_transform])

    def _apply(t: Union[BaseTransform, Iterable],
               func: Callable[[BaseTransform], None]):
        # Note that BaseTransform and Iterable are not mutually exclusive,
        # e.g. Compose, RandomChoice
        if isinstance(t, BaseTransform):
            if hasattr(t, '_methods_with_randomness'):
                func(t)
        if isinstance(t, Iterable):
            for _t in t:
                _apply(_t, func)

    try:
        _apply(transforms, _start_cache)
        yield
    finally:
        _apply(transforms, _end_cache)
