# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch

from mmcv.runner import set_random_seed
from mmcv.utils import TORCH_VERSION, digit_version

is_rocm_pytorch = False
if digit_version(TORCH_VERSION) >= digit_version('1.5'):
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and
                               (ROCM_HOME is not None)) else False


def test_set_random_seed():
    set_random_seed(0)
    a_random = random.randint(0, 10)
    a_np_random = np.random.rand(2, 2)
    a_torch_random = torch.rand(2, 2)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False
    assert os.environ['PYTHONHASHSEED'] == str(0)

    set_random_seed(0, True)
    b_random = random.randint(0, 10)
    b_np_random = np.random.rand(2, 2)
    b_torch_random = torch.rand(2, 2)
    assert torch.backends.cudnn.deterministic is True
    if is_rocm_pytorch:
        assert torch.backends.cudnn.benchmark is True
    else:
        assert torch.backends.cudnn.benchmark is False

    assert a_random == b_random
    assert np.equal(a_np_random, b_np_random).all()
    assert torch.equal(a_torch_random, b_torch_random)
