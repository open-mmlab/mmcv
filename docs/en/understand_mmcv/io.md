## File IO

This module provides two universal API to load and dump files of different formats.

```{note}
Since v1.3.16, the IO modules support loading (dumping) data from (to) different backends, respectively. More details are in PR [#1330](https://github.com/open-mmlab/mmcv/pull/1330).
```

### Load and dump data

`mmcv` provides a universal api for loading and dumping data, currently
supported formats are json, yaml and pickle.

#### Load from disk or dump to disk

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

#### Load from other backends or dump to other backends

```python
import mmcv

# load data from a file
data = mmcv.load('s3://bucket-name/test.json')
data = mmcv.load('s3://bucket-name/test.yaml')
data = mmcv.load('s3://bucket-name/test.pkl')

# dump data to a file with a filename (infer format from file extension)
mmcv.dump(data, 's3://bucket-name/out.pkl')
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

#### Load from disk

Use `list_from_file` to load the list from a.txt.

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

For example `b.txt` is a text file with 3 lines.

```
1 cat
2 dog cow
3 panda
```

Then use `dict_from_file` to load the dict from `b.txt`.

```python
>>> mmcv.dict_from_file('b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

#### Load from other backends

Use `list_from_file` to load the list from `s3://bucket-name/a.txt`.

```python
>>> mmcv.list_from_file('s3://bucket-name/a.txt')
['a', 'b', 'c', 'd', 'e']
>>> mmcv.list_from_file('s3://bucket-name/a.txt', offset=2)
['c', 'd', 'e']
>>> mmcv.list_from_file('s3://bucket-name/a.txt', max_num=2)
['a', 'b']
>>> mmcv.list_from_file('s3://bucket-name/a.txt', prefix='/mnt/')
['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

Use `dict_from_file` to load the dict from `s3://bucket-name/b.txt`.

```python
>>> mmcv.dict_from_file('s3://bucket-name/b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('s3://bucket-name/b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

### Load and dump checkpoints

#### Load checkpoints from disk or save to disk

We can read the checkpoints from disk or save to disk in the following way.

```python
import torch

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = '/path/of/your/checkpoint2.pth'
# read from filepath1
checkpoint = torch.load(filepath1)
# save to filepath2
torch.save(checkpoint, filepath2)
```

MMCV provides many backends. `HardDiskBackend` is one of them and we can use it to read or save checkpoints.

```python
import io
from mmcv.fileio.file_client import HardDiskBackend

disk_backend = HardDiskBackend()
with io.BytesIO(disk_backend.get(filepath1)) as buffer:
    checkpoint = torch.load(buffer)
with io.BytesIO() as buffer:
    torch.save(checkpoint, buffer)
    disk_backend.put(buffer.getvalue(), filepath2)
```

If we want to implement an interface which automatically select the corresponding
backend based on the file path, we can use the `FileClient`.
For example, we want to implement two methods for reading checkpoints as well as saving checkpoints,
which need to support different types of file paths, either disk paths, network paths or other paths.

```python
from mmcv.fileio.file_client import FileClient

def load_checkpoint(path):
    file_client = FileClient.infer(uri=path)
    with io.BytesIO(file_client.get(path)) as buffer:
        checkpoint = torch.load(buffer)
    return checkpoint

def save_checkpoint(checkpoint, path):
    with io.BytesIO() as buffer:
        torch.save(checkpoint, buffer)
        file_client.put(buffer.getvalue(), path)

file_client = FileClient.infer_client(uri=filepath1)
checkpoint = load_checkpoint(filepath1)
save_checkpoint(checkpoint, filepath2)
```

#### Load checkpoints from the Internet

```{note}
Currently, it only supports reading checkpoints from the Internet, and does not support saving checkpoints to the Internet.
```

```python
import io
import torch
from mmcv.fileio.file_client import HTTPBackend, FileClient

filepath = 'http://path/of/your/checkpoint.pth'
checkpoint = torch.utils.model_zoo.load_url(filepath)

http_backend = HTTPBackend()
with io.BytesIO(http_backend.get(filepath)) as buffer:
    checkpoint = torch.load(buffer)

file_client = FileClient.infer_client(uri=filepath)
with io.BytesIO(file_client.get(filepath)) as buffer:
    checkpoint = torch.load(buffer)
```
