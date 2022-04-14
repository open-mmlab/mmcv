# Copyright (c) OpenMMLab. All rights reserved.


def is_ipu():
    try:
        import poptorch  # noqa: E261, F401
    except ImportError:
        return False
    return True


IS_IPU = is_ipu()
