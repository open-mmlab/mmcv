# Copyright (c) OpenMMLab. All rights reserved.
import pytest

import mmcv
from mmcv import deprecated_api_warning
from mmcv.utils.misc import has_method


def test_to_ntuple():
    single_number = 2
    assert mmcv.utils.to_1tuple(single_number) == (single_number, )
    assert mmcv.utils.to_2tuple(single_number) == (single_number,
                                                   single_number)
    assert mmcv.utils.to_3tuple(single_number) == (single_number,
                                                   single_number,
                                                   single_number)
    assert mmcv.utils.to_4tuple(single_number) == (single_number,
                                                   single_number,
                                                   single_number,
                                                   single_number)
    assert mmcv.utils.to_ntuple(5)(single_number) == (single_number,
                                                      single_number,
                                                      single_number,
                                                      single_number,
                                                      single_number)
    assert mmcv.utils.to_ntuple(6)(single_number) == (single_number,
                                                      single_number,
                                                      single_number,
                                                      single_number,
                                                      single_number,
                                                      single_number)


def test_iter_cast():
    assert mmcv.list_cast([1, 2, 3], int) == [1, 2, 3]
    assert mmcv.list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert mmcv.list_cast([1, 2, 3], str) == ['1', '2', '3']
    assert mmcv.tuple_cast((1, 2, 3), str) == ('1', '2', '3')
    assert next(mmcv.iter_cast([1, 2, 3], str)) == '1'
    with pytest.raises(TypeError):
        mmcv.iter_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        mmcv.iter_cast(1, str)


def test_is_seq_of():
    assert mmcv.is_seq_of([1.0, 2.0, 3.0], float)
    assert mmcv.is_seq_of([(1, ), (2, ), (3, )], tuple)
    assert mmcv.is_seq_of((1.0, 2.0, 3.0), float)
    assert mmcv.is_list_of([1.0, 2.0, 3.0], float)
    assert not mmcv.is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not mmcv.is_tuple_of([1.0, 2.0, 3.0], float)
    assert not mmcv.is_seq_of([1.0, 2, 3], int)
    assert not mmcv.is_seq_of((1.0, 2, 3), int)


def test_slice_list():
    in_list = [1, 2, 3, 4, 5, 6]
    assert mmcv.slice_list(in_list, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert mmcv.slice_list(in_list, [len(in_list)]) == [in_list]
    with pytest.raises(TypeError):
        mmcv.slice_list(in_list, 2.0)
    with pytest.raises(ValueError):
        mmcv.slice_list(in_list, [1, 2])


def test_concat_list():
    assert mmcv.concat_list([[1, 2]]) == [1, 2]
    assert mmcv.concat_list([[1, 2], [3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]


def test_requires_package(capsys):

    @mmcv.requires_package('nnn')
    def func_a():
        pass

    @mmcv.requires_package(['numpy', 'n1', 'n2'])
    def func_b():
        pass

    @mmcv.requires_package('numpy')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_requires_executable(capsys):

    @mmcv.requires_executable('nnn')
    def func_a():
        pass

    @mmcv.requires_executable(['ls', 'n1', 'n2'])
    def func_b():
        pass

    @mmcv.requires_executable('mv')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_import_modules_from_strings():
    # multiple imports
    import os.path as osp_
    import sys as sys_
    osp, sys = mmcv.import_modules_from_strings(['os.path', 'sys'])
    assert osp == osp_
    assert sys == sys_

    # single imports
    osp = mmcv.import_modules_from_strings('os.path')
    assert osp == osp_
    # No imports
    assert mmcv.import_modules_from_strings(None) is None
    assert mmcv.import_modules_from_strings([]) is None
    assert mmcv.import_modules_from_strings('') is None
    # Unsupported types
    with pytest.raises(TypeError):
        mmcv.import_modules_from_strings(1)
    with pytest.raises(TypeError):
        mmcv.import_modules_from_strings([1])
    # Failed imports
    with pytest.raises(ImportError):
        mmcv.import_modules_from_strings('_not_implemented_module')
    with pytest.warns(UserWarning):
        imported = mmcv.import_modules_from_strings(
            '_not_implemented_module', allow_failed_imports=True)
        assert imported is None
    with pytest.warns(UserWarning):
        imported = mmcv.import_modules_from_strings(
            ['os.path', '_not_implemented'], allow_failed_imports=True)
        assert imported[0] == osp
        assert imported[1] is None


def test_is_method_overridden():

    class Base:

        def foo1():
            pass

        def foo2():
            pass

    class Sub(Base):

        def foo1():
            pass

    # test passing sub class directly
    assert mmcv.is_method_overridden('foo1', Base, Sub)
    assert not mmcv.is_method_overridden('foo2', Base, Sub)

    # test passing instance of sub class
    sub_instance = Sub()
    assert mmcv.is_method_overridden('foo1', Base, sub_instance)
    assert not mmcv.is_method_overridden('foo2', Base, sub_instance)

    # base_class should be a class, not instance
    base_instance = Base()
    with pytest.raises(AssertionError):
        mmcv.is_method_overridden('foo1', base_instance, sub_instance)


def test_has_method():

    class Foo:

        def __init__(self, name):
            self.name = name

        def print_name(self):
            print(self.name)

    foo = Foo('foo')
    assert not has_method(foo, 'name')
    assert has_method(foo, 'print_name')


def test_deprecated_api_warning():

    @deprecated_api_warning(name_dict=dict(old_key='new_key'))
    def dummy_func(new_key=1):
        return new_key

    # replace `old_key` to `new_key`
    assert dummy_func(old_key=2) == 2

    # The expected behavior is to replace the
    # deprecated key `old_key` to `new_key`,
    # but got them in the arguments at the same time
    with pytest.raises(AssertionError):
        dummy_func(old_key=1, new_key=2)
