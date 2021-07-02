## File IO

This module provides two universal API to load and dump files of different formats.

### Load and dump data

`mmcv` provides a universal api for loading and dumping data, currently
supported formats are json, yaml and pickle.

```python
import mmcv

# load data from a file
data = mmcv.load('test.json')
data = mmcv.load('test.yaml')
data = mmcv.load('test.pkl')
# load data from a file-like object
with open('test.json', 'r') as f:
    data = mmcv.load(f, file_format='json')

# dump data to a string
json_str = mmcv.dump(data, file_format='json')

# dump data to a file with a filename (infer format from file extension)
mmcv.dump(data, 'out.pkl')

# dump data to a file with a file-like object
with open('test.yaml', 'w') as f:
    data = mmcv.dump(data, f, file_format='yaml')
```

It is also very convenient to extend the api to support more file formats.
All you need to do is to write a file handler inherited from `BaseFileHandler`
and register it with one or several file formats.

You need to implement at least 3 methods.

```python
import mmcv

# To register multiple file formats, a list can be used as the argument.
# @mmcv.register_handler(['txt', 'log'])
@mmcv.register_handler('txt')
class TxtHandler1(mmcv.BaseFileHandler):

    def load_from_fileobj(self, file):
        return file.read()

    def dump_to_fileobj(self, obj, file):
        file.write(str(obj))

    def dump_to_str(self, obj, **kwargs):
        return str(obj)
```

Here is an example of `PickleHandler`.

```python
import pickle

class PickleHandler(mmcv.BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(
            obj, filepath, mode='wb', **kwargs)
```

### Load a text file as a list or dict

For example `a.txt` is a text file with 5 lines.

```
a
b
c
d
e
```

Then use `list_from_file` to load the list from a.txt.

```python
>>> mmcv.list_from_file('a.txt')
['a', 'b', 'c', 'd', 'e']
>>> mmcv.list_from_file('a.txt', offset=2)
['c', 'd', 'e']
>>> mmcv.list_from_file('a.txt', max_num=2)
['a', 'b']
>>> mmcv.list_from_file('a.txt', prefix='/mnt/')
['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

For example `b.txt` is a text file with 5 lines.

```
1 cat
2 dog cow
3 panda
```

Then use `dict_from_file` to load the list from a.txt.

```python
>>> mmcv.dict_from_file('b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```
