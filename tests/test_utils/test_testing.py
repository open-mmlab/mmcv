import pytest
import torch.nn as nn

import mmcv

check_dict_data_1 = [({'a': 'test1', 'b': 2, 'c': (4, 6)}, ('a', 'b', 'c'))]
check_dict_data_2 = [
    (('test1', 2, (4, 6)), True), (('test1', 2, (6, 4)), False),
    (('test1', 2, None), False), (('test2', 2, (4, 6)), False)
]


@pytest.mark.parametrize('result_dict, key_list', check_dict_data_1)
@pytest.mark.parametrize('value_list, ret_value', check_dict_data_2)
def test_check_dict(result_dict, key_list, value_list, ret_value):
    assert mmcv.check_dict(result_dict, key_list, value_list) == ret_value


def test_check_class_attr():

    class TestExample(object):
        a, b, c = 1, ('wvi', 3), [4.5, 3.14]

        def test_func(self):
            return self.b

    assert mmcv.check_class_attr(TestExample, ('a', 'b', 'c'),
                                 (1, ('wvi', 3), [4.5, 3.14]))
    assert not mmcv.check_class_attr(TestExample, ('a', 'b', 'c'),
                                     (1, ('wvi', 3), [4.5, 3.14, 2]))
    assert not mmcv.check_class_attr(TestExample, ('bc', 'c'),
                                     (54, [4.5, 3.14]))
    assert mmcv.check_class_attr(TestExample, ('b', 'test_func'),
                                 (('wvi', 3), TestExample.test_func))


check_keys_contain_data_1 = [(['res_layer', 'norm_layer', 'dense_layer'])]
check_keys_contain_data_2 = [(['res_layer', 'dense_layer'], True),
                             (['res_layer', 'conv_layer'], False)]


@pytest.mark.parametrize('result_keys', check_keys_contain_data_1)
@pytest.mark.parametrize('target_keys, ret_value', check_keys_contain_data_2)
def test_check_keys_contain(result_keys, target_keys, ret_value):
    assert mmcv.check_keys_contain(result_keys, target_keys) == ret_value


check_keys_equal_data_1 = [(['res_layer', 'norm_layer', 'dense_layer'])]
check_keys_equal_data_2 = [(['res_layer', 'norm_layer', 'dense_layer'], True),
                           (['res_layer', 'dense_layer', 'norm_layer'], True),
                           (['res_layer', 'norm_layer'], False),
                           (['res_layer', 'conv_layer', 'norm_layer'], False)]


@pytest.mark.parametrize('result_keys', check_keys_equal_data_1)
@pytest.mark.parametrize('target_keys, ret_value', check_keys_equal_data_2)
def test_check_keys_equal(result_keys, target_keys, ret_value):
    assert mmcv.check_keys_equal(result_keys, target_keys) == ret_value


check_norm_state_data = [((True, True, True, True, True), True, True),
                         ((False, True, True, True, True), True, True),
                         ((True, True, True, False, True), True, False),
                         ((True, False, True, False, True), False, True)]


@pytest.mark.parametrize('modules_state, train_state, ret_value',
                         check_norm_state_data)
def test_check_norm_state(modules_state, train_state, ret_value):
    demo_modules = nn.Sequential(*[
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, 3),
        nn.BatchNorm2d(64),
        nn.Softmax()
    ])
    for demo_module, state in zip(demo_modules, modules_state):
        demo_module.train(state)
    assert mmcv.check_norm_state(demo_modules, train_state) == ret_value


is_block_data = [(nn.Conv3d(3, 64, 3), (mmcv._ConvNd), True),
                 (nn.Conv2d(3, 64, 3), (nn.Conv1d, nn.Conv3d), False),
                 (nn.Linear(512, 10), (nn.Linear, nn.Identity), True),
                 (nn.Sigmoid(), (nn.Softmax), False)]


@pytest.mark.parametrize('module, block_candidates, ret_value', is_block_data)
def test_is_block(module, block_candidates, ret_value):
    assert mmcv.is_block(module, block_candidates) == ret_value


is_norm_data = [(nn.Conv3d(3, 64, 3), False), (nn.BatchNorm3d(128), True),
                (nn.GroupNorm(8, 64), True), (nn.Sigmoid(), False)]


@pytest.mark.parametrize('module, ret_value', is_norm_data)
def test_is_norm(module, ret_value):
    assert mmcv.is_norm(module) == ret_value


def test_is_all_zeros():
    demo_module = nn.Conv2d(3, 64, 3)
    nn.init.constant_(demo_module.weight, 0)
    nn.init.constant_(demo_module.bias, 0)
    assert mmcv.is_all_zeros(demo_module)

    nn.init.xavier_normal_(demo_module.weight)
    nn.init.constant_(demo_module.bias, 0)
    assert not mmcv.is_all_zeros(demo_module)

    demo_module = nn.Linear(2048, 400, bias=False)
    nn.init.constant_(demo_module.weight, 0)
    assert mmcv.is_all_zeros(demo_module)

    nn.init.normal_(demo_module.weight, mean=0, std=0.01)
    assert not mmcv.is_all_zeros(demo_module)
