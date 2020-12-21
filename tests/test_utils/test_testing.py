import pytest

import mmcv

check_dict_unittest_data = [({
    'a': 'test1',
    'b': 2,
    'c': (4, 6)
}, ('a', 'b', 'c'), ('test1', 2, (4, 6)), True),
                            ({
                                'a': 'test1',
                                'b': 2,
                                'c': (6, 4)
                            }, ('a', 'b', 'c'), ('test1', 2, (4, 6)), False),
                            ({
                                'a': 'test1',
                                'b': [2],
                                'c': (4, 6)
                            }, ('a', 'b', 'c'), ('test1', 2, (4, 6)), False),
                            ({
                                'a': 'test1',
                                'b': 2,
                                'c': (4, 6)
                            }, ('a', 'b', 'c'), ('test2', 2, (4, 6)), False)]


@pytest.mark.parametrize('result_dict, key_list, value_list, ret_value',
                         check_dict_unittest_data)
def test_check_dict(result_dict, key_list, value_list, ret_value):
    assert mmcv.check_dict(result_dict, key_list, value_list) == ret_value
