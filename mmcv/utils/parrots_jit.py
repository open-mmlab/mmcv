import pytest
import torch

TORCH_VERSION = torch.__version__

if TORCH_VERSION == 'parrots':
    from parrots.jit import pat as jit
else:

    def jit(func=None,
            check_input=None,
            partial_shape=False,
            derivate=False,
            coderize=False,
            optimize=False):

        def wrapper(func):

            def wrapper_inner(*args, **kargs):
                return func(*args, **kargs)

            return wrapper_inner

        if func is None:
            return wrapper
        else:
            return func


if TORCH_VERSION == 'parrots':
    from parrots.utils.tester import skip_no_elena
else:

    def bypass_decorator(func):

        def wrapper(*args, **kargs):
            return func(*args, **kargs)

        return wrapper

    skip_no_elena = bypass_decorator


def is_using_parrots():
    return TORCH_VERSION == 'parrots'


skip_no_parrots = pytest.mark.skipif(
    not is_using_parrots(), reason='test case under parrots environment')
