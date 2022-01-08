import pytest
import torch

from mmcv.ops import get_compiler_version, get_compiling_cuda_version


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires CUDA support')
class TestInfo(object):

    def test_info(self):
        cv = get_compiler_version()
        ccv = get_compiling_cuda_version()
        assert cv is not None
        assert ccv is not None
