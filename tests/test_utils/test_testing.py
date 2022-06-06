# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

import mmcv

try:
    import torch
except ImportError:
    torch = None
else:
    import torch.nn as nn


def test_assert_dict_contains_subset():
    dict_obj = {'a': 'test1', 'b': 2, 'c': (4, 6)}

    # case 1
    expected_subset = {'a': 'test1', 'b': 2, 'c': (4, 6)}
    assert mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    # case 2
    expected_subset = {'a': 'test1', 'b': 2, 'c': (6, 4)}
    assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    # case 3
    expected_subset = {'a': 'test1', 'b': 2, 'c': None}
    assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    # case 4
    expected_subset = {'a': 'test1', 'b': 2, 'd': (4, 6)}
    assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    # case 5
    dict_obj = {
        'a': 'test1',
        'b': 2,
        'c': (4, 6),
        'd': np.array([[5, 3, 5], [1, 2, 3]])
    }
    expected_subset = {
        'a': 'test1',
        'b': 2,
        'c': (4, 6),
        'd': np.array([[5, 3, 5], [6, 2, 3]])
    }
    assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    # case 6
    dict_obj = {'a': 'test1', 'b': 2, 'c': (4, 6), 'd': np.array([[1]])}
    expected_subset = {'a': 'test1', 'b': 2, 'c': (4, 6), 'd': np.array([[1]])}
    assert mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

    if torch is not None:
        dict_obj = {
            'a': 'test1',
            'b': 2,
            'c': (4, 6),
            'd': torch.tensor([5, 3, 5])
        }

        # case 7
        expected_subset = {'d': torch.tensor([5, 5, 5])}
        assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)

        # case 8
        expected_subset = {'d': torch.tensor([[5, 3, 5], [4, 1, 2]])}
        assert not mmcv.assert_dict_contains_subset(dict_obj, expected_subset)


def test_assert_attrs_equal():

    class TestExample:
        a, b, c = 1, ('wvi', 3), [4.5, 3.14]

        def test_func(self):
            return self.b

    # case 1
    assert mmcv.assert_attrs_equal(TestExample, {
        'a': 1,
        'b': ('wvi', 3),
        'c': [4.5, 3.14]
    })

    # case 2
    assert not mmcv.assert_attrs_equal(TestExample, {
        'a': 1,
        'b': ('wvi', 3),
        'c': [4.5, 3.14, 2]
    })

    # case 3
    assert not mmcv.assert_attrs_equal(TestExample, {
        'bc': 54,
        'c': [4.5, 3.14]
    })

    # case 4
    assert mmcv.assert_attrs_equal(TestExample, {
        'b': ('wvi', 3),
        'test_func': TestExample.test_func
    })

    if torch is not None:

        class TestExample:
            a, b = torch.tensor([1]), torch.tensor([4, 5])

        # case 5
        assert mmcv.assert_attrs_equal(TestExample, {
            'a': torch.tensor([1]),
            'b': torch.tensor([4, 5])
        })

        # case 6
        assert not mmcv.assert_attrs_equal(TestExample, {
            'a': torch.tensor([1]),
            'b': torch.tensor([4, 6])
        })


assert_dict_has_keys_data_1 = [({
    'res_layer': 1,
    'norm_layer': 2,
    'dense_layer': 3
})]
assert_dict_has_keys_data_2 = [(['res_layer', 'dense_layer'], True),
                               (['res_layer', 'conv_layer'], False)]


@pytest.mark.parametrize('obj', assert_dict_has_keys_data_1)
@pytest.mark.parametrize('expected_keys, ret_value',
                         assert_dict_has_keys_data_2)
def test_assert_dict_has_keys(obj, expected_keys, ret_value):
    assert mmcv.assert_dict_has_keys(obj, expected_keys) == ret_value


assert_keys_equal_data_1 = [(['res_layer', 'norm_layer', 'dense_layer'])]
assert_keys_equal_data_2 = [(['res_layer', 'norm_layer', 'dense_layer'], True),
                            (['res_layer', 'dense_layer', 'norm_layer'], True),
                            (['res_layer', 'norm_layer'], False),
                            (['res_layer', 'conv_layer', 'norm_layer'], False)]


@pytest.mark.parametrize('result_keys', assert_keys_equal_data_1)
@pytest.mark.parametrize('target_keys, ret_value', assert_keys_equal_data_2)
def test_assert_keys_equal(result_keys, target_keys, ret_value):
    assert mmcv.assert_keys_equal(result_keys, target_keys) == ret_value


@pytest.mark.skipif(torch is None, reason='requires torch library')
def test_assert_is_norm_layer():
    # case 1
    assert not mmcv.assert_is_norm_layer(nn.Conv3d(3, 64, 3))

    # case 2
    assert mmcv.assert_is_norm_layer(nn.BatchNorm3d(128))

    # case 3
    assert mmcv.assert_is_norm_layer(nn.GroupNorm(8, 64))

    # case 4
    assert not mmcv.assert_is_norm_layer(nn.Sigmoid())


@pytest.mark.skipif(torch is None, reason='requires torch library')
def test_assert_params_all_zeros():
    demo_module = nn.Conv2d(3, 64, 3)
    nn.init.constant_(demo_module.weight, 0)
    nn.init.constant_(demo_module.bias, 0)
    assert mmcv.assert_params_all_zeros(demo_module)

    nn.init.xavier_normal_(demo_module.weight)
    nn.init.constant_(demo_module.bias, 0)
    assert not mmcv.assert_params_all_zeros(demo_module)

    demo_module = nn.Linear(2048, 400, bias=False)
    nn.init.constant_(demo_module.weight, 0)
    assert mmcv.assert_params_all_zeros(demo_module)

    nn.init.normal_(demo_module.weight, mean=0, std=0.01)
    assert not mmcv.assert_params_all_zeros(demo_module)


def test_check_python_script(capsys):
    mmcv.utils.check_python_script('./tests/data/scripts/hello.py zz')
    captured = capsys.readouterr().out
    assert captured == 'hello zz!\n'
    mmcv.utils.check_python_script('./tests/data/scripts/hello.py agent')
    captured = capsys.readouterr().out
    assert captured == 'hello agent!\n'
    # Make sure that wrong cmd raises an error
    with pytest.raises(SystemExit):
        mmcv.utils.check_python_script('./tests/data/scripts/hello.py li zz')
