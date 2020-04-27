# Copyright (c) Open-MMLab. All rights reserved.
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
    folder = osp.join(osp.dirname(__file__), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']
    assert set(mmcv.scandir(folder)) == set(filenames)
    assert set(mmcv.scandir(Path(folder))) == set(filenames)
    assert set(mmcv.scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(mmcv.scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(mmcv.scandir(folder, '.png')) == set()

    filenames_recursive = [
        'a.bin', '1.txt', '2.txt', '1.json', '2.json', 'sub/1.json',
        'sub/1.txt'
    ]
    assert set(mmcv.scandir(folder,
                            recursive=True)) == set(filenames_recursive)
    assert set(mmcv.scandir(Path(folder),
                            recursive=True)) == set(filenames_recursive)
    assert set(mmcv.scandir(folder, '.txt', recursive=True)) == set([
        filename for filename in filenames_recursive
        if filename.endswith('.txt')
    ])
    with pytest.raises(TypeError):
        list(mmcv.scandir(123))
    with pytest.raises(TypeError):
        list(mmcv.scandir(folder, 111))
