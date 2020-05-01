# Copyright (c) Open-MMLab. All rights reserved.
import argparse
import json
import os.path as osp
import sys
import tempfile

import pytest

from mmcv import Config, DictAction


def test_construct():
    cfg = Config()
    assert cfg.filename is None
    assert cfg.text == ''
    assert len(cfg) == 0
    assert cfg._cfg_dict == {}

    with pytest.raises(TypeError):
        Config([0, 1])

    cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4='test')
    format_text = json.dumps(cfg_dict, indent=2)
    for filename in ['a.py', 'b.json', 'c.yaml']:
        cfg_file = osp.join(osp.dirname(__file__), 'data/config', filename)
        cfg = Config(cfg_dict, filename=cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == open(cfg_file, 'r').read()
        if sys.version_info >= (3, 6):
            assert cfg.dump() == format_text
        else:
            loaded = json.loads(cfg.dump())
            assert set(loaded.keys()) == set(cfg_dict)


def test_fromfile():
    for filename in ['a.py', 'a.b.py', 'b.json', 'c.yaml']:
        cfg_file = osp.join(osp.dirname(__file__), 'data/config', filename)
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == osp.abspath(osp.expanduser(cfg_file)) + '\n' + \
            open(cfg_file, 'r').read()

    with pytest.raises(FileNotFoundError):
        Config.fromfile('no_such_file.py')
    with pytest.raises(IOError):
        Config.fromfile(osp.join(osp.dirname(__file__), 'data/color.jpg'))


def test_merge_from_base():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/d.py')
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    base_cfg_file = osp.join(osp.dirname(__file__), 'data/config/base.py')
    merge_text = osp.abspath(osp.expanduser(base_cfg_file)) + '\n' + \
        open(base_cfg_file, 'r').read()
    merge_text += '\n' + osp.abspath(osp.expanduser(cfg_file)) + '\n' + \
                  open(cfg_file, 'r').read()
    assert cfg.text == merge_text
    assert cfg.item1 == [2, 3]
    assert cfg.item2.a == 1
    assert cfg.item3 is False
    assert cfg.item4 == 'test_base'

    with pytest.raises(TypeError):
        Config.fromfile(osp.join(osp.dirname(__file__), 'data/config/e.py'))


def test_merge_from_multiple_bases():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/l.py')
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    # cfg.field
    assert cfg.item1 == [1, 2]
    assert cfg.item2.a == 0
    assert cfg.item3 is False
    assert cfg.item4 == 'test'
    assert cfg.item5 == dict(a=0, b=1)
    assert cfg.item6 == [dict(a=0), dict(b=1)]
    assert cfg.item7 == dict(a=[0, 1, 2], b=dict(c=[3.1, 4.2, 5.3]))

    with pytest.raises(KeyError):
        Config.fromfile(osp.join(osp.dirname(__file__), 'data/config/m.py'))


def test_merge_recursive_bases():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/f.py')
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    # cfg.field
    assert cfg.item1 == [2, 3]
    assert cfg.item2.a == 1
    assert cfg.item3 is False
    assert cfg.item4 == 'test_recursive_bases'


def test_merge_from_dict():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/a.py')
    cfg = Config.fromfile(cfg_file)
    input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}
    cfg.merge_from_dict(input_options)
    assert cfg.item2 == dict(a=1, b=0.1)
    assert cfg.item3 is False


def test_merge_delete():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/delete.py')
    cfg = Config.fromfile(cfg_file)
    # cfg.field
    assert cfg.item1 == [1, 2]
    assert cfg.item2 == dict(b=0)
    assert cfg.item3 is True
    assert cfg.item4 == 'test'
    assert '_delete_' not in cfg.item2


def test_merge_intermediate_variable():

    cfg_file = osp.join(osp.dirname(__file__), 'data/config/i_child.py')
    cfg = Config.fromfile(cfg_file)
    # cfg.field
    assert cfg.item1 == [1, 2]
    assert cfg.item2 == dict(a=0)
    assert cfg.item3 is True
    assert cfg.item4 == 'test'
    assert cfg.item_cfg == dict(b=2)
    assert cfg.item5 == dict(cfg=dict(b=1))
    assert cfg.item6 == dict(cfg=dict(b=2))


def test_fromfile_in_config():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/code.py')
    cfg = Config.fromfile(cfg_file)
    # cfg.field
    assert cfg.cfg.item1 == [1, 2]
    assert cfg.cfg.item2 == dict(a=0)
    assert cfg.cfg.item3 is True
    assert cfg.cfg.item4 == 'test'
    assert cfg.item5 == 1


def test_dict():
    cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4='test')

    for filename in ['a.py', 'b.json', 'c.yaml']:
        cfg_file = osp.join(osp.dirname(__file__), 'data/config', filename)
        cfg = Config.fromfile(cfg_file)

        # len(cfg)
        assert len(cfg) == 4
        # cfg.keys()
        assert set(cfg.keys()) == set(cfg_dict.keys())
        assert set(cfg._cfg_dict.keys()) == set(cfg_dict.keys())
        # cfg.values()
        for value in cfg.values():
            assert value in cfg_dict.values()
        # cfg.items()
        for name, value in cfg.items():
            assert name in cfg_dict
            assert value in cfg_dict.values()
        # cfg.field
        assert cfg.item1 == cfg_dict['item1']
        assert cfg.item2 == cfg_dict['item2']
        assert cfg.item2.a == 0
        assert cfg.item3 == cfg_dict['item3']
        assert cfg.item4 == cfg_dict['item4']
        with pytest.raises(AttributeError):
            cfg.not_exist
        # field in cfg, cfg[field], cfg.get()
        for name in ['item1', 'item2', 'item3', 'item4']:
            assert name in cfg
            assert cfg[name] == cfg_dict[name]
            assert cfg.get(name) == cfg_dict[name]
            assert cfg.get('not_exist') is None
            assert cfg.get('not_exist', 0) == 0
            with pytest.raises(KeyError):
                cfg['not_exist']
        assert 'item1' in cfg
        assert 'not_exist' not in cfg
        # cfg.update()
        cfg.update(dict(item1=0))
        assert cfg.item1 == 0
        cfg.update(dict(item2=dict(a=1)))
        assert cfg.item2.a == 1


def test_setattr():
    cfg = Config()
    cfg.item1 = [1, 2]
    cfg.item2 = {'a': 0}
    cfg['item5'] = {'a': {'b': None}}
    assert cfg._cfg_dict['item1'] == [1, 2]
    assert cfg.item1 == [1, 2]
    assert cfg._cfg_dict['item2'] == {'a': 0}
    assert cfg.item2.a == 0
    assert cfg._cfg_dict['item5'] == {'a': {'b': None}}
    assert cfg.item5.a.b is None


def test_pretty_text():
    cfg_file = osp.join(osp.dirname(__file__), 'data/config/l.py')
    cfg = Config.fromfile(cfg_file)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        text_cfg_filename = osp.join(temp_config_dir, '_text_config.py')
        with open(text_cfg_filename, 'w') as f:
            f.write(cfg.pretty_text)
        text_cfg = Config.fromfile(text_cfg_filename)
    assert text_cfg._cfg_dict == cfg._cfg_dict


def test_dict_action():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args(
        ['--options', 'item2.a=1', 'item2.b=0.1', 'item2.c=x', 'item3=false'])
    out_dict = {'item2.a': 1, 'item2.b': 0.1, 'item2.c': 'x', 'item3': False}
    assert args.options == out_dict

    cfg_file = osp.join(osp.dirname(__file__), 'data/config/a.py')
    cfg = Config.fromfile(cfg_file)
    cfg.merge_from_dict(args.options)
    assert cfg.item2 == dict(a=1, b=0.1, c='x')
    assert cfg.item3 is False
