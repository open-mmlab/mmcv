## Config

`Config` class is used for manipulating config and config files. It supports
loading configs from multiple file formats including **python**, **json** and **yaml**.
It provides dict-like apis to get and set values.

Here is an example of the config file `test.py`.

```python
a = 1
b = dict(b1=[0, 1, 2], b2=None)
c = (1, 2)
d = 'string'
```

To load and use configs

```python
>>> cfg = Config.fromfile('test.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b1=[0, 1, 2], b2=None),
...      c=(1, 2),
...      d='string')
```

For all format configs, some predefined variables are supported. It will convert the variable in `{{ var }}` with its real value.

Currently, it supports four predefined variables:

`{{ fileDirname }}` - the current opened file's dirname, e.g. /home/your-username/your-project/folder

`{{ fileBasename }}` - the current opened file's basename, e.g. file.ext

`{{ fileBasenameNoExtension }}` - the current opened file's basename with no file extension, e.g. file

`{{ fileExtname }}` - the current opened file's extension, e.g. .ext

These variable names are referred from [VS Code](https://code.visualstudio.com/docs/editor/variables-reference).

Here is one examples of config with predefined variables.

`config_a.py`

```python
a = 1
b = './work_dir/{{ fileBasenameNoExtension }}'
c = '{{ fileExtname }}'
```

```python
>>> cfg = Config.fromfile('./config_a.py')
>>> print(cfg)
>>> dict(a=1,
...      b='./work_dir/config_a',
...      c='.py')
```

For all format configs, inheritance is supported. To reuse fields in other config files,
specify `_base_='./config_a.py'` or a list of configs `_base_=['./config_a.py', './config_b.py']`.
Here are 4 examples of config inheritance.

`config_a.py`

```python
a = 1
b = dict(b1=[0, 1, 2], b2=None)
```

### Inherit from base config without overlapped keys

`config_b.py`

```python
_base_ = './config_a.py'
c = (1, 2)
d = 'string'
```

```python
>>> cfg = Config.fromfile('./config_b.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b1=[0, 1, 2], b2=None),
...      c=(1, 2),
...      d='string')
```

New fields in `config_b.py` are combined with old fields in `config_a.py`

### Inherit from base config with overlapped keys

`config_c.py`

```python
_base_ = './config_a.py'
b = dict(b2=1)
c = (1, 2)
```

```python
>>> cfg = Config.fromfile('./config_c.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b1=[0, 1, 2], b2=1),
...      c=(1, 2))
```

`b.b2=None` in `config_a` is replaced with `b.b2=1` in `config_c.py`.

### Inherit from base config with ignored fields

`config_d.py`

```python
_base_ = './config_a.py'
b = dict(_delete_=True, b2=None, b3=0.1)
c = (1, 2)
```

```python
>>> cfg = Config.fromfile('./config_d.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b2=None, b3=0.1),
...      c=(1, 2))
```

You may also set `_delete_=True` to ignore some fields in base configs. All old keys `b1, b2, b3` in `b` are replaced with new keys `b2, b3`.

### Inherit from multiple base configs (the base configs should not contain the same keys)

`config_e.py`

```python
c = (1, 2)
d = 'string'
```

`config_f.py`

```python
_base_ = ['./config_a.py', './config_e.py']
```

```python
>>> cfg = Config.fromfile('./config_f.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b1=[0, 1, 2], b2=None),
...      c=(1, 2),
...      d='string')
```

### Reference variables from base

You can reference variables defined in base using the following grammar.

`base.py`

```python
item1 = 'a'
item2 = dict(item3 = 'b')
```

`config_g.py`

```python
_base_ = ['./base.py']
item = dict(a = {{ _base_.item1 }}, b = {{ _base_.item2.item3 }})
```

```python
>>> cfg = Config.fromfile('./config_g.py')
>>> print(cfg.pretty_text)
item1 = 'a'
item2 = dict(item3='b')
item = dict(a='a', b='b')
```

### Add deprecation information in configs

Deprecation information can be added in a config file, which will trigger a `UserWarning` when this config file is loaded.

`deprecated_cfg.py`

```python
_base_ = 'expected_cfg.py'

_deprecation_ = dict(
    expected = 'expected_cfg.py',  # optional to show expected config path in the warning information
    reference = 'url to related PR'  # optional to show reference link in the warning information
)
```

```python
>>> cfg = Config.fromfile('./deprecated_cfg.py')

UserWarning: The config file deprecated_cfg.py will be deprecated in the future. Please use expected_cfg.py instead. More information can be found at https://github.com/open-mmlab/mmcv/pull/1275
```
