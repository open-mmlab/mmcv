# Copyright (c) OpenMMLab. All rights reserved.
import torch

TORCH_VERSION = torch.__version__


def is_cuda() -> bool:
    return torch.cuda.is_available()


def is_mlu() -> bool:
    if TORCH_VERSION != 'parrots':
        try:
            return torch.is_mlu_available()
        except AttributeError:
            return False
    return False
