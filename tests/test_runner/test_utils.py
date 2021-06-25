import os
import random

import numpy as np
import torch

from mmcv.runner import set_random_seed


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
    assert torch.backends.cudnn.benchmark is False

    assert a_random == b_random
    assert np.equal(a_np_random, b_np_random).all()
    assert torch.equal(a_torch_random, b_torch_random)
