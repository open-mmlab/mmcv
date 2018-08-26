import os
import os.path as osp
import tempfile

import mmcv
import pytest


def _test_processor(file_format, test_obj, str_checker, mode='r+'):
    # dump to a string
    dump_str = mmcv.dump(test_obj, file_format=file_format)
    str_checker(dump_str)

    # load/dump with filenames
    tmp_filename = osp.join(tempfile.gettempdir(), 'mmcv_test_dump')
    mmcv.dump(test_obj, tmp_filename, file_format=file_format)
    assert osp.isfile(tmp_filename)
    load_obj = mmcv.load(tmp_filename, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # json load/dump with a file-like object
    with tempfile.NamedTemporaryFile(mode, delete=False) as f:
        tmp_filename = f.name
        mmcv.dump(test_obj, f, file_format=file_format)
    assert osp.isfile(tmp_filename)
    with open(tmp_filename, mode) as f:
        load_obj = mmcv.load(f, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # automatically inference the file format from the given filename
    tmp_filename = osp.join(tempfile.gettempdir(),
                            'mmcv_test_dump.' + file_format)
    mmcv.dump(test_obj, tmp_filename)
    assert osp.isfile(tmp_filename)
    load_obj = mmcv.load(tmp_filename)
    assert load_obj == test_obj
    os.remove(tmp_filename)


obj_for_test = [{'a': 'abc', 'b': 1}, 2, 'c']


def test_json():

    def json_checker(dump_str):
        assert dump_str in [
            '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
        ]

    _test_processor('json', obj_for_test, json_checker)


def test_yaml():

    def yaml_checker(dump_str):
        assert dump_str in [
            '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n'
        ]

    _test_processor('yaml', obj_for_test, yaml_checker)


def test_pickle():

    def pickle_checker(dump_str):
        import pickle
        assert pickle.loads(dump_str) == obj_for_test

    _test_processor('pickle', obj_for_test, pickle_checker, mode='rb+')


def test_exception():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']

    with pytest.raises(ValueError):
        mmcv.dump(test_obj)

    with pytest.raises(TypeError):
        mmcv.dump(test_obj, 'tmp.txt')
