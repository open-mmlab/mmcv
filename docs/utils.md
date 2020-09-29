## Utils

### Config

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

#### Inherit from base config without overlaped keys

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

#### Inherit from base config with overlaped keys

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

#### Inherit from base config with ignored fields

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

#### Inherit from multiple base configs (the base configs should not contain the same keys)

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

### ProgressBar

If you want to apply a method to a list of items and track the progress, `track_progress`
is a good choice. It will display a progress bar to tell the progress and ETA.

```python
import mmcv

def func(item):
    # do something
    pass

tasks = [item_1, item_2, ..., item_n]

mmcv.track_progress(func, tasks)
```

The output is like the following.
![progress](_static/progress.gif)

There is another method `track_parallel_progress`, which wraps multiprocessing and
progress visualization.

```python
mmcv.track_parallel_progress(func, tasks, 8)  # 8 workers
```

![progress](_static/parallel_progress.gif)

If you want to iterate or enumerate a list of items and track the progress, `track_iter_progress`
is a good choice. It will display a progress bar to tell the progress and ETA.

```python
import mmcv

tasks = [item_1, item_2, ..., item_n]

for task in mmcv.track_iter_progress(tasks):
    # do something like print
    print(task)

for i, task in enumerate(mmcv.track_iter_progress(tasks)):
    # do something like print
    print(i)
    print(task)
```

### Timer

It is convinient to compute the runtime of a code block with `Timer`.

```python
import time

with mmcv.Timer():
    # simulate some code block
    time.sleep(1)
```

or try with `since_start()` and `since_last_check()`. This former can
return the runtime since the timer starts and the latter will return the time
since the last time checked.

```python
timer = mmcv.Timer()
# code block 1 here
print(timer.since_start())
# code block 2 here
print(timer.since_last_check())
print(timer.since_start())
```
