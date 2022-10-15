# Copyright (c) OpenMMLab. All rights reserved.
import torch
if 'parrots' != torch.__version__:
    from mmengine.device import (is_cuda_available, is_mlu_available,
                                is_mps_available)

    IS_MLU_AVAILABLE = is_mlu_available()
    IS_MPS_AVAILABLE = is_mps_available()
    IS_CUDA_AVAILABLE = is_cuda_available()
    IS_CAMB_AVAILABLE = False
else:
    from parrots.base import use_cuda, use_camb
    IS_MLU_AVAILABLE = False
    IS_MPS_AVAILABLE = False
    IS_CUDA_AVAILABLE = torch.cuda.is_available() and use_cuda
    IS_CAMB_AVAILABLE = torch.cuda.is_available() and use_camb 


