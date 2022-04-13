# Copyright (c) OpenMMLab. All rights reserved.


def is_ipu():
    try:
        import poptorch  # noqa: E261, F401
        IS_IPU = True
    except ImportError:
        IS_IPU = False
    return IS_IPU


IS_IPU = is_ipu()
