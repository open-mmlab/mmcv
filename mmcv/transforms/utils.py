# Copyright (c) OpenMMLab. All rights reserved.

import copy
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
        Only an instance method can be decorated with ``cache_randomness``.
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

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)

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
        return copy.deepcopy(self)


def avoid_cache_randomness(cls):
    """Decorator that marks a data transform class (subclass of
    :class:`BaseTransform`) prohibited from caching randomness. With this
    decorator, errors will be raised in following cases:

        1. A method is defined in the class with the decorate
    `cache_randomness`;
        2. An instance of the class is invoked with the context
    `cache_random_params`.

    A typical usage of `avoid_cache_randomness` is to decorate the data
    transforms with non-cacheable random behaviors (e.g., the random behavior
    can not be defined in a method, thus can not be decorated with
    `cache_randomness`). This is for preventing unintentinoal use of such data
    transforms within the context of caching randomness, which may lead to
    unexpected results.
    """

    # Check that cls is a data transform class
    assert issubclass(cls, BaseTransform)

    # Check that no method is decorated with `cache_randomness` in cls
    if getattr(cls, '_methods_with_randomness', None):
        raise RuntimeError(
            f'Class {cls.__name__} decorated with '
            '``avoid_cache_randomness`` should not have methods decorated '
            'with ``cache_randomness`` (invalid methods: '
            f'{cls._methods_with_randomness})')

    class AvoidCacheRandomness:

        def __get__(self, obj, objtype=None):
            # Here we check the value in `objtype.__dict__` instead of
            # directly checking the attribute
            # `objtype._avoid_cache_randomness`. So if the base class is
            # decorated with :func:`avoid_cache_randomness`, it will not be
            # inherited by subclasses.
            return objtype.__dict__.get('_avoid_cache_randomness', False)

    cls.avoid_cache_randomness = AvoidCacheRandomness()
    cls._avoid_cache_randomness = True

    return cls


@contextmanager
def cache_random_params(transforms: Union[BaseTransform, Iterable]):
    """Context-manager that enables the cache of return values of methods
    decorated with ``cache_randomness`` in transforms.

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
    key2counter: dict = defaultdict(int)

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
                        'The method decorated with ``cache_randomness`` '
                        'should be invoked at most once during processing '
                        f'one data sample. The method {name} of {obj} has '
                        f'been invoked {key2counter[key]} times.')
            return output

        return wrapped

    def _start_cache(t: BaseTransform):
        # Check if cache is allowed for `t`
        if getattr(t, 'avoid_cache_randomness', False):
            raise RuntimeError(
                f'Class {t.__class__.__name__} decorated with '
                '``avoid_cache_randomness`` is not allowed to be used with'
                ' ``cache_random_params`` (e.g. wrapped by '
                '``ApplyToMultiple`` with ``share_random_params==True``).')

        # Skip transforms w/o random method
        if not hasattr(t, '_methods_with_randomness'):
            return

        # Set cache enabled flag
        setattr(t, '_cache_enabled', True)

        # Store the original method and init the counter
        if hasattr(t, '_methods_with_randomness'):
            setattr(t, 'transform', _add_invoke_checker(t, 'transform'))
            for name in getattr(t, '_methods_with_randomness'):
                setattr(t, name, _add_invoke_counter(t, name))

    def _end_cache(t: BaseTransform):
        # Skip transforms w/o random method
        if not hasattr(t, '_methods_with_randomness'):
            return

        # Remove cache enabled flag
        delattr(t, '_cache_enabled')
        if hasattr(t, '_cache'):
            delattr(t, '_cache')

        # Restore the original method
        if hasattr(t, '_methods_with_randomness'):
            for name in getattr(t, '_methods_with_randomness'):
                key = f'{id(t)}.{name}'
                setattr(t, name, key2method[key])

            key_transform = f'{id(t)}.transform'
            setattr(t, 'transform', key2method[key_transform])

    def _apply(t: Union[BaseTransform, Iterable],
               func: Callable[[BaseTransform], None]):
        if isinstance(t, BaseTransform):
            func(t)
        if isinstance(t, Iterable):
            for _t in t:
                _apply(_t, func)

    try:
        _apply(transforms, _start_cache)
        yield
    finally:
        _apply(transforms, _end_cache)
