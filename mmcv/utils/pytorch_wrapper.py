# Copyright (c) OpenMMLab. All rights reserved.
import torch

TORCH_VERSION = torch.__version__


def is_cuda() -> bool:
    return torch.cuda.is_available()
