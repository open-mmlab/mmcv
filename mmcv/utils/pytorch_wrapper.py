# Copyright (c) OpenMMLab. All rights reserved.
import torch

TORCH_VERSION = torch.__version__


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


IS_CUDA_AVAILABLE = is_cuda_available()
