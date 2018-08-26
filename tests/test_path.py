import os.path as osp
import sys

import mmcv
import pytest


def test_list_from_file():
    filename = osp.join(osp.dirname(__file__), 'data/filelist.txt')
    filelist = mmcv.list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = mmcv.list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = mmcv.list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = mmcv.list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = mmcv.list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']


def test_dict_from_file():
    filename = osp.join(osp.dirname(__file__), 'data/mapping.txt')
    mapping = mmcv.dict_from_file(filename)
    assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
    mapping = mmcv.dict_from_file(filename, key_type=int)
    assert mapping == {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}


def test_check_file_exist():
    mmcv.check_file_exist(__file__)
    if sys.version_info > (3, 3):
        with pytest.raises(FileNotFoundError):
            mmcv.check_file_exist('no_such_file.txt')
    else:
        with pytest.raises(IOError):
            mmcv.check_file_exist('no_such_file.txt')


def test_scandir():
    folder = osp.join(osp.dirname(__file__), 'data/for_scan')
    filenames = ['a.bin', '1.txt', '2.txt', '1.json', '2.json']

    assert set(mmcv.scandir(folder)) == set(filenames)
    assert set(mmcv.scandir(folder, '.txt')) == set(
        [filename for filename in filenames if filename.endswith('.txt')])
    assert set(mmcv.scandir(folder, ('.json', '.txt'))) == set([
        filename for filename in filenames
        if filename.endswith(('.txt', '.json'))
    ])
    assert set(mmcv.scandir(folder, '.png')) == set()
    with pytest.raises(TypeError):
        mmcv.scandir(folder, 111)
