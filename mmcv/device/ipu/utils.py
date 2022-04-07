# Copyright (c) OpenMMLab. All rights reserved.


def get_ipu_mode():
    try:
        import poptorch # noqa: E261, F401
        IPU_MODE = True
    except ImportError:
        IPU_MODE = False
    return IPU_MODE


IPU_MODE = get_ipu_mode()
