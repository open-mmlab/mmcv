# Copyright (c) Open-MMLab. All rights reserved.
import pytest

import mmcv


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


def test_custom_imports():
    # multiple imports
    runner, image = mmcv.import_modules_from_strings(
        ['mmcv.runner', 'mmcv.image'])
    import mmcv.runner as runner2
    import mmcv.image as image2
    assert runner == runner2
    assert image == image2
    # single imports
    runner = mmcv.import_modules_from_strings('mmcv.runner')
    assert runner == runner2
    # No imports
    assert mmcv.import_modules_from_strings(None) is None
    assert mmcv.import_modules_from_strings([]) is None
    assert mmcv.import_modules_from_strings('') is None
    # Unsupported types
    with pytest.raises(TypeError):
        mmcv.import_modules_from_strings(1)
    with pytest.raises(TypeError):
        mmcv.import_modules_from_strings([1])
