import pytest
import torch

import mmcv
from mmcv.utils import TORCH_VERSION

skip_no_parrots = pytest.mark.skipif(
    TORCH_VERSION != 'parrots', reason='test case under parrots environment')


class TestJit(object):

    def test_add_dict(self):

        @mmcv.jit
        def add_dict(oper):
            rets = oper['x'] + oper['y']
            return {'result': rets}

        def add_dict_pyfunc(oper):
            rets = oper['x'] + oper['y']
            return {'result': rets}

        a = torch.rand((3, 4))
        b = torch.rand((3, 4))
        oper = {'x': a, 'y': b}

        rets_t = add_dict(oper)
        rets = add_dict_pyfunc(oper)
        assert 'result' in rets
        assert (rets_t['result'] == rets['result']).all()

    def test_add_list(self):

        @mmcv.jit
        def add_list(oper, x, y):
            rets = {}
            for idx, pair in enumerate(oper):
                rets[f'k{idx}'] = pair['x'] + pair['y']
            rets[f'k{len(oper)}'] = x + y
            return rets

        def add_list_pyfunc(oper, x, y):
            rets = {}
            for idx, pair in enumerate(oper):
                rets[f'k{idx}'] = pair['x'] + pair['y']
            rets[f'k{len(oper)}'] = x + y
            return rets

        pair_num = 3
        oper = []
        for _ in range(pair_num):
            oper.append({'x': torch.rand((3, 4)), 'y': torch.rand((3, 4))})
        a = torch.rand((3, 4))
        b = torch.rand((3, 4))
        rets = add_list_pyfunc(oper, x=a, y=b)
        rets_t = add_list(oper, x=a, y=b)
        for idx in range(pair_num + 1):
            assert f'k{idx}' in rets_t
            assert (rets[f'k{idx}'] == rets_t[f'k{idx}']).all()

    @skip_no_parrots
    def test_jit_cache(self):

        @mmcv.jit
        def func(oper):
            if oper['const'] > 1:
                return oper['x'] * 2 + oper['y']
            else:
                return oper['x'] * 2 - oper['y']

        def pyfunc(oper):
            if oper['const'] > 1:
                return oper['x'] * 2 + oper['y']
            else:
                return oper['x'] * 2 - oper['y']

        assert len(func._cache._cache) == 0

        oper = {'const': 2, 'x': torch.rand((3, 4)), 'y': torch.rand((3, 4))}
        rets_plus = pyfunc(oper)
        rets_plus_t = func(oper)
        assert (rets_plus == rets_plus_t).all()
        assert len(func._cache._cache) == 1

        oper['const'] = 0.5
        rets_minus = pyfunc(oper)
        rets_minus_t = func(oper)
        assert (rets_minus == rets_minus_t).all()
        assert len(func._cache._cache) == 2

        rets_a = (rets_minus_t + rets_plus_t) / 4
        assert torch.allclose(oper['x'], rets_a)

    @skip_no_parrots
    def test_jit_shape(self):

        @mmcv.jit
        def func(a):
            return a + 1

        assert len(func._cache._cache) == 0

        a = torch.ones((3, 4))
        r = func(a)
        assert r.shape == (3, 4)
        assert (r == 2).all()
        assert len(func._cache._cache) == 1

        a = torch.ones((2, 3, 4))
        r = func(a)
        assert r.shape == (2, 3, 4)
        assert (r == 2).all()
        assert len(func._cache._cache) == 2

    @skip_no_parrots
    def test_jit_kwargs(self):

        @mmcv.jit
        def func(a, b):
            return torch.mean((a - b) * (a - b))

        assert len(func._cache._cache) == 0
        x = torch.rand((16, 32))
        y = torch.rand((16, 32))
        func(x, y)
        assert len(func._cache._cache) == 1
        func(x, b=y)
        assert len(func._cache._cache) == 1
        func(b=y, a=x)
        assert len(func._cache._cache) == 1

    def test_jit_derivate(self):

        @mmcv.jit(derivate=True)
        def func(x, y):
            return (x + 2) * (y - 2)

        a = torch.rand((3, 4))
        b = torch.rand((3, 4))
        a.requires_grad = True

        c = func(a, b)
        assert c.requires_grad
        d = torch.empty_like(c)
        d.fill_(1.0)
        c.backward(d)
        assert torch.allclose(a.grad, (b - 2))
        assert b.grad is None

        a.grad = None
        c = func(a, b)
        assert c.requires_grad
        d = torch.empty_like(c)
        d.fill_(2.7)
        c.backward(d)
        assert torch.allclose(a.grad, 2.7 * (b - 2))
        assert b.grad is None

    def test_jit_optimize(self):

        @mmcv.jit(optimize=True)
        def func(a, b):
            return torch.mean((a - b) * (a - b))

        def pyfunc(a, b):
            return torch.mean((a - b) * (a - b))

        a = torch.rand((16, 32))
        b = torch.rand((16, 32))

        c = func(a, b)
        d = pyfunc(a, b)
        assert torch.allclose(c, d)

    @mmcv.skip_no_elena
    def test_jit_coderize(self):
        if not torch.cuda.is_available():
            return

        @mmcv.jit(coderize=True)
        def func(a, b):
            return (a + b) * (a - b)

        def pyfunc(a, b):
            return (a + b) * (a - b)

        a = torch.rand((16, 32), device='cuda')
        b = torch.rand((16, 32), device='cuda')

        c = func(a, b)
        d = pyfunc(a, b)
        assert torch.allclose(c, d)

    def test_jit_value_dependent(self):

        @mmcv.jit
        def func(a, b):
            torch.nonzero(a)
            return torch.mean((a - b) * (a - b))

        def pyfunc(a, b):
            torch.nonzero(a)
            return torch.mean((a - b) * (a - b))

        a = torch.rand((16, 32))
        b = torch.rand((16, 32))

        c = func(a, b)
        d = pyfunc(a, b)
        assert torch.allclose(c, d)

    @skip_no_parrots
    def test_jit_check_input(self):

        def func(x):
            y = torch.rand_like(x)
            return x + y

        a = torch.ones((3, 4))
        with pytest.raises(AssertionError):
            func = mmcv.jit(func, check_input=(a, ))

    @skip_no_parrots
    def test_jit_partial_shape(self):

        @mmcv.jit(full_shape=False)
        def func(a, b):
            return torch.mean((a - b) * (a - b))

        def pyfunc(a, b):
            return torch.mean((a - b) * (a - b))

        a = torch.rand((3, 4))
        b = torch.rand((3, 4))
        assert torch.allclose(func(a, b), pyfunc(a, b))
        assert len(func._cache._cache) == 1

        a = torch.rand((6, 5))
        b = torch.rand((6, 5))
        assert torch.allclose(func(a, b), pyfunc(a, b))
        assert len(func._cache._cache) == 1

        a = torch.rand((3, 4, 5))
        b = torch.rand((3, 4, 5))
        assert torch.allclose(func(a, b), pyfunc(a, b))
        assert len(func._cache._cache) == 2

        a = torch.rand((1, 9, 8))
        b = torch.rand((1, 9, 8))
        assert torch.allclose(func(a, b), pyfunc(a, b))
        assert len(func._cache._cache) == 2

    def test_instance_method(self):

        class T(object):

            def __init__(self, shape):
                self._c = torch.rand(shape)

            @mmcv.jit
            def test_method(self, x, y):
                return (x * self._c) + y

        shape = (16, 32)
        t = T(shape)
        a = torch.rand(shape)
        b = torch.rand(shape)
        res = (a * t._c) + b
        jit_res = t.test_method(a, b)
        assert torch.allclose(res, jit_res)

        t = T(shape)
        res = (a * t._c) + b
        jit_res = t.test_method(a, b)
        assert torch.allclose(res, jit_res)
