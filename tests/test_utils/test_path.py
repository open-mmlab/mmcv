# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path

import pytest

import mmcv


def test_is_filepath():
    assert mmcv.is_filepath(__file__)
    assert mmcv.is_filepath('abc')
    assert mmcv.is_filepath(Path('/etc'))
    assert not mmcv.is_filepath(0)


def test_fopen():
    assert hasattr(mmcv.fopen(__file__), 'read')
    assert hasattr(mmcv.fopen(Path(__file__)), 'read')


def test_check_file_exist():
    mmcv.check_file_exist(__file__)
    with pytest.raises(FileNotFoundError):
        mmcv.check_file_exist('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(osp.dirname(__file__)), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json', '3.TXT']
    assert set(mmcv.scandir(folder)) == set(filenames)
    assert set(mmcv.scandir(Path(folder))) == set(filenames)
    assert set(mmcv.scandir(folder, '.txt')) == {
        filename
        for filename in filenames if filename.endswith('.txt')
    }
    assert set(mmcv.scandir(folder, ('.json', '.txt'))) == {
        filename
        for filename in filenames if filename.endswith(('.txt', '.json'))
    }
    assert set(mmcv.scandir(folder, '.png')) == set()

    # path of sep is `\\` in windows but `/` in linux, so osp.join should be
    # used to join string for compatibility
    filenames_recursive = [
        'a.bin', '1.txt', '2.txt', '1.json', '2.json', '3.TXT',
        osp.join('sub', '1.json'),
        osp.join('sub', '1.txt'), '.file'
    ]
    # .file starts with '.' and is a file so it will not be scanned
    assert set(mmcv.scandir(folder, recursive=True)) == {
        filename
        for filename in filenames_recursive if filename != '.file'
    }
    assert set(mmcv.scandir(Path(folder), recursive=True)) == {
        filename
        for filename in filenames_recursive if filename != '.file'
    }
    assert set(mmcv.scandir(folder, '.txt', recursive=True)) == {
        filename
        for filename in filenames_recursive if filename.endswith('.txt')
    }
    assert set(
        mmcv.scandir(folder, '.TXT', recursive=True,
                     case_sensitive=False)) == {
                         filename
                         for filename in filenames_recursive
                         if filename.endswith(('.txt', '.TXT'))
                     }
    assert set(
        mmcv.scandir(
            folder, ('.TXT', '.JSON'), recursive=True,
            case_sensitive=False)) == {
                filename
                for filename in filenames_recursive
                if filename.endswith(('.txt', '.json', '.TXT'))
            }
    with pytest.raises(TypeError):
        list(mmcv.scandir(123))
    with pytest.raises(TypeError):
        list(mmcv.scandir(folder, 111))
