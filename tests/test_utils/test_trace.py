from distutils.version import LooseVersion

import pytest
import torch

from mmcv.utils import is_jit_tracing


@pytest.mark.skipif(
    LooseVersion(torch.__version__) < LooseVersion('1.6.0'),
    reason='torch.jit.is_tracing is not available before 1.6.0')
def test_is_jit_tracing():

    def foo(x):
        if is_jit_tracing():
            return x
        else:
            return x.tolist()

    x = torch.rand(3)
    # test without trace
    assert isinstance(foo(x), list)

    # test with trace
    traced_foo = torch.jit.trace(foo, (torch.rand(1), ))
    assert isinstance(traced_foo(x), torch.Tensor)
