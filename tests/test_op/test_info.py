import torch

from mmcv.op import get_compiler_version, get_compiling_cuda_version


class TestInfo(object):

    def test_info(self):
        if not torch.cuda.is_available():
            return
        cv = get_compiler_version()
        ccv = get_compiling_cuda_version()
        assert cv is not None
        assert ccv is not None
