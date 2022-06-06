# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.utils import model_zoo

from mmcv.utils import TORCH_VERSION, digit_version, load_url


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not necessary in parrots test')
def test_load_url():
    url1 = 'https://download.openmmlab.com/mmcv/test_data/saved_in_pt1.5.pth'
    url2 = 'https://download.openmmlab.com/mmcv/test_data/saved_in_pt1.6.pth'

    # The 1.6 release of PyTorch switched torch.save to use a new zipfile-based
    # file format. It will cause RuntimeError when a checkpoint was saved in
    # torch >= 1.6.0 but loaded in torch < 1.7.0.
    # More details at https://github.com/open-mmlab/mmpose/issues/904
    if digit_version(TORCH_VERSION) < digit_version('1.7.0'):
        model_zoo.load_url(url1)
        with pytest.raises(RuntimeError):
            model_zoo.load_url(url2)
    else:
        # high version of PyTorch can load checkpoints from url, regardless
        # of which version they were saved in
        model_zoo.load_url(url1)
        model_zoo.load_url(url2)

    load_url(url1)
    # if a checkpoint was saved in torch >= 1.6.0 but loaded in torch < 1.5.0,
    # it will raise a RuntimeError
    if digit_version(TORCH_VERSION) < digit_version('1.5.0'):
        with pytest.raises(RuntimeError):
            load_url(url2)
    else:
        load_url(url2)
