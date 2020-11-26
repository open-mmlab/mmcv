import torch

TORCH_VERSION = torch.__version__


if TORCH_VERSION == 'parrots':
    from parrots.jit import pat as jit
else:
    def pat(func=None, check_input=None, full_shape=True,
            derivate=False, coderize=False, optimize=False):
        def wrapper(func):
            def wraper_inner(*args, **kargs):
                return func(*args, **kargs)
            return wrapper_inner
        if func is None:
            return wrapper
        else:
            return func


if TORCH_VERSION == 'parrots':
    from parrots.utils.tester import skip_no_cuda, skip_no_elena
else:
    def bypass_decorator(func):
        def wraper(*args, **kargs):
            return func(*args, **kargs)
        return wrapper

    skip_no_cuda = bypass_decorator
    skip_no_elena = bypass_decorator
