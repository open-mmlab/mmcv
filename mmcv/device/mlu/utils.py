# Copyright (c) OpenMMLab. All rights reserved.
def is_mlu_available():
    try:
        import torch
        return (hasattr(torch, 'is_mlu_available')
                and torch.is_mlu_available())
    except Exception:
        return False


IS_MLU_AVAILABLE = is_mlu_available()
