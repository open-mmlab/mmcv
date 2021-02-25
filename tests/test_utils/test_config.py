# Copyright (c) Open-MMLab. All rights reserved.
import argparse
import json
import os
import os.path as osp
import shutil
import tempfile

import pytest
import yaml

from mmcv import Config, DictAction, dump, load

data_path = osp.join(osp.dirname(osp.dirname(__file__)), 'data')


def test_construct():
    cfg = Config()
    assert cfg.filename is None
    assert cfg.text == ''
    assert len(cfg) == 0
    assert cfg._cfg_dict == {}

    with pytest.raises(TypeError):
        Config([0, 1])

    cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4='test')
    # test a.py
    cfg_file = osp.join(data_path, 'config/a.py')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == cfg.pretty_text
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'a.py')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)

    # test b.json
    cfg_file = osp.join(data_path, 'config/b.json')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == json.dumps(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'b.json')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)

    # test c.yaml
    cfg_file = osp.join(data_path, 'config/c.yaml')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == yaml.dump(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'c.yaml')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)

    # test h.py
    cfg_file = osp.join(data_path, 'config/h.py')
    cfg_dict = dict(
        item1='h.py',
        item2=f'{osp.dirname(__file__)}/data/config',
        item3='abc_h')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == cfg.pretty_text
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'h.py')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']
        assert Config.fromfile(dump_file)['item2'] == cfg_dict['item2']
        assert Config.fromfile(dump_file)['item3'] == cfg_dict['item3']

    # test no use_predefined_variable
    cfg_dict = dict(
        item1='{{fileBasename}}',
        item2='{{ fileDirname}}',
        item3='abc_{{ fileBasenameNoExtension }}')
    assert Config.fromfile(cfg_file, False)
    assert Config.fromfile(cfg_file, False)['item1'] == cfg_dict['item1']
    assert Config.fromfile(cfg_file, False)['item2'] == cfg_dict['item2']
    assert Config.fromfile(cfg_file, False)['item3'] == cfg_dict['item3']

    # test p.yaml
    cfg_file = osp.join(data_path, 'config/p.yaml')
    cfg_dict = dict(item1=f'{osp.dirname(__file__)}/data/config')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == yaml.dump(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'p.yaml')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']

    # test no use_predefined_variable
    assert Config.fromfile(cfg_file, False)
    assert Config.fromfile(cfg_file, False)['item1'] == '{{ fileDirname }}'

    # test o.json
    cfg_file = osp.join(data_path, 'config/o.json')
    cfg_dict = dict(item1=f'{osp.dirname(__file__)}/data/config')
    cfg = Config(cfg_dict, filename=cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    assert cfg.text == open(cfg_file, 'r').read()
    assert cfg.dump() == json.dumps(cfg_dict)
    with tempfile.TemporaryDirectory() as temp_config_dir:
        dump_file = osp.join(temp_config_dir, 'o.json')
        cfg.dump(dump_file)
        assert cfg.dump() == open(dump_file, 'r').read()
        assert Config.fromfile(dump_file)
        assert Config.fromfile(dump_file)['item1'] == cfg_dict['item1']

    # test no use_predefined_variable
    assert Config.fromfile(cfg_file, False)
    assert Config.fromfile(cfg_file, False)['item1'] == '{{ fileDirname }}'


def test_fromfile():
    for filename in ['a.py', 'a.b.py', 'b.json', 'c.yaml']:
        cfg_file = osp.join(data_path, 'config', filename)
        cfg = Config.fromfile(cfg_file)
        assert isinstance(cfg, Config)
        assert cfg.filename == cfg_file
        assert cfg.text == osp.abspath(osp.expanduser(cfg_file)) + '\n' + \
            open(cfg_file, 'r').read()

    # test custom_imports for Config.fromfile
    cfg_file = osp.join(data_path, 'config', 'q.py')
    imported_file = osp.join(data_path, 'config', 'r.py')
    target_pkg = osp.join(osp.dirname(__file__), 'r.py')

    # Since the imported config will be regarded as a tmp file
    # it should be copied to the directory at the same level
    shutil.copy(imported_file, target_pkg)
    Config.fromfile(cfg_file, import_custom_modules=True)

    assert os.environ.pop('TEST_VALUE') == 'test'
    os.remove(target_pkg)

    with pytest.raises(FileNotFoundError):
        Config.fromfile('no_such_file.py')
    with pytest.raises(IOError):
        Config.fromfile(osp.join(data_path, 'color.jpg'))


def test_fromstring():
    for filename in ['a.py', 'a.b.py', 'b.json', 'c.yaml']:
        cfg_file = osp.join(data_path, 'config', filename)
        file_format = osp.splitext(filename)[-1]
        in_cfg = Config.fromfile(cfg_file)

        out_cfg = Config.fromstring(in_cfg.pretty_text, '.py')
        assert in_cfg._cfg_dict == out_cfg._cfg_dict

        cfg_str = open(cfg_file, 'r').read()
        out_cfg = Config.fromstring(cfg_str, file_format)
        assert in_cfg._cfg_dict == out_cfg._cfg_dict

    # test pretty_text only supports py file format
    cfg_file = osp.join(data_path, 'config', 'b.json')
    in_cfg = Config.fromfile(cfg_file)
    with pytest.raises(Exception):
        Config.fromstring(in_cfg.pretty_text, '.json')

    # test file format error
    cfg_str = open(cfg_file, 'r').read()
    with pytest.raises(Exception):
        Config.fromstring(cfg_str, '.py')


def test_merge_from_base():
    cfg_file = osp.join(data_path, 'config/d.py')
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    base_cfg_file = osp.join(data_path, 'config/base.py')
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
        Config.fromfile(osp.join(data_path, 'config/e.py'))


def test_merge_from_multiple_bases():
    cfg_file = osp.join(data_path, 'config/l.py')
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
        Config.fromfile(osp.join(data_path, 'config/m.py'))


def test_merge_recursive_bases():
    cfg_file = osp.join(data_path, 'config/f.py')
    cfg = Config.fromfile(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.filename == cfg_file
    # cfg.field
    assert cfg.item1 == [2, 3]
    assert cfg.item2.a == 1
    assert cfg.item3 is False
    assert cfg.item4 == 'test_recursive_bases'


def test_merge_from_dict():
    cfg_file = osp.join(data_path, 'config/a.py')
    cfg = Config.fromfile(cfg_file)
    input_options = {'item2.a': 1, 'item2.b': 0.1, 'item3': False}
    cfg.merge_from_dict(input_options)
    assert cfg.item2 == dict(a=1, b=0.1)
    assert cfg.item3 is False

    cfg_file = osp.join(data_path, 'config/s.py')
    cfg = Config.fromfile(cfg_file)

    # Allow list keys
    input_options = {'item.0.a': 1, 'item.1.b': 1}
    cfg.merge_from_dict(input_options, allow_list_keys=True)
    assert cfg.item == [{'a': 1}, {'b': 1, 'c': 0}]

    # allow_list_keys is False
    input_options = {'item.0.a': 1, 'item.1.b': 1}
    with pytest.raises(TypeError):
        cfg.merge_from_dict(input_options, allow_list_keys=False)

    # Overflowed index number
    input_options = {'item.2.a': 1}
    with pytest.raises(KeyError):
        cfg.merge_from_dict(input_options, allow_list_keys=True)


def test_merge_delete():
    cfg_file = osp.join(data_path, 'config/delete.py')
    cfg = Config.fromfile(cfg_file)
    # cfg.field
    assert cfg.item1 == [1, 2]
    assert cfg.item2 == dict(b=0)
    assert cfg.item3 is True
    assert cfg.item4 == 'test'
    assert '_delete_' not in cfg.item2


def test_merge_intermediate_variable():

    cfg_file = osp.join(data_path, 'config/i_child.py')
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
    cfg_file = osp.join(data_path, 'config/code.py')
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
        cfg_file = osp.join(data_path, 'config', filename)
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
    cfg_file = osp.join(data_path, 'config/l.py')
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
    # Nested brackets
    args = parser.parse_args(
        ['--options', 'item2.a=a,b', 'item2.b=[(a,b), [1,2], false]'])
    out_dict = {'item2.a': ['a', 'b'], 'item2.b': [('a', 'b'), [1, 2], False]}
    assert args.options == out_dict
    # Single Nested brackets
    args = parser.parse_args(['--options', 'item2.a=[[1]]'])
    out_dict = {'item2.a': [[1]]}
    assert args.options == out_dict
    # Imbalance bracket
    with pytest.raises(AssertionError):
        parser.parse_args(['--options', 'item2.a=[(a,b), [1,2], false'])
    # Normal values
    args = parser.parse_args(
        ['--options', 'item2.a=1', 'item2.b=0.1', 'item2.c=x', 'item3=false'])
    out_dict = {'item2.a': 1, 'item2.b': 0.1, 'item2.c': 'x', 'item3': False}
    assert args.options == out_dict
    cfg_file = osp.join(data_path, 'config/a.py')
    cfg = Config.fromfile(cfg_file)
    cfg.merge_from_dict(args.options)
    assert cfg.item2 == dict(a=1, b=0.1, c='x')
    assert cfg.item3 is False


def test_dump_mapping():
    cfg_file = osp.join(data_path, 'config/n.py')
    cfg = Config.fromfile(cfg_file)

    with tempfile.TemporaryDirectory() as temp_config_dir:
        text_cfg_filename = osp.join(temp_config_dir, '_text_config.py')
        cfg.dump(text_cfg_filename)
        text_cfg = Config.fromfile(text_cfg_filename)

    assert text_cfg._cfg_dict == cfg._cfg_dict


def test_reserved_key():
    cfg_file = osp.join(data_path, 'config/g.py')
    with pytest.raises(KeyError):
        Config.fromfile(cfg_file)


def test_syntax_error():
    temp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py')
    temp_cfg_path = temp_cfg_file.name
    # write a file with syntax error
    with open(temp_cfg_path, 'w') as f:
        f.write('a=0b=dict(c=1)')
    with pytest.raises(
            SyntaxError,
            match='There are syntax errors in config '
            f'file {temp_cfg_path}'):
        Config.fromfile(temp_cfg_path)
    temp_cfg_file.close()


def test_pickle_support():
    cfg_file = osp.join(data_path, 'config/n.py')
    cfg = Config.fromfile(cfg_file)

    with tempfile.TemporaryDirectory() as temp_config_dir:
        pkl_cfg_filename = osp.join(temp_config_dir, '_pickle.pkl')
        dump(cfg, pkl_cfg_filename)
        pkl_cfg = load(pkl_cfg_filename)

    assert pkl_cfg._cfg_dict == cfg._cfg_dict
