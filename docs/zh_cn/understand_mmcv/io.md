## 文件输入输出

文件输入输出模块提供了两个通用的 API 接口用于读取和保存不同格式的文件。

```{note}
在 v1.3.16 及之后的版本中，IO 模块支持从不同后端读取数据并支持将数据至不同后端。更多细节请访问 PR [#1330](https://github.com/open-mmlab/mmcv/pull/1330)。
```

### 读取和保存数据

`mmcv` 提供了一个通用的 api 用于读取和保存数据，目前支持的格式有 json、yaml 和 pickle。

#### 从硬盘读取数据或者将数据保存至硬盘

```python
import mmcv

# 从文件中读取数据
data = mmcv.load('test.json')
data = mmcv.load('test.yaml')
data = mmcv.load('test.pkl')
# 从文件对象中读取数据
with open('test.json', 'r') as f:
    data = mmcv.load(f, file_format='json')

# 将数据序列化为字符串
json_str = mmcv.dump(data, file_format='json')

# 将数据保存至文件 (根据文件名后缀反推文件类型)
mmcv.dump(data, 'out.pkl')

# 将数据保存至文件对象
with open('test.yaml', 'w') as f:
    data = mmcv.dump(data, f, file_format='yaml')
```

#### 从其他后端加载或者保存至其他后端

```python
import mmcv

# 从 s3 文件读取数据
data = mmcv.load('s3://bucket-name/test.json')
data = mmcv.load('s3://bucket-name/test.yaml')
data = mmcv.load('s3://bucket-name/test.pkl')

# 将数据保存至 s3 文件 (根据文件名后缀反推文件类型)
mmcv.dump(data, 's3://bucket-name/out.pkl')
```

我们提供了易于拓展的方式以支持更多的文件格式。我们只需要创建一个继承自 `BaseFileHandler` 的
文件句柄类并将其注册到 `mmcv` 中即可。句柄类至少需要重写三个方法。

```python
import mmcv

# 支持为文件句柄类注册多个文件格式
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

以 `PickleHandler` 为例

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

### 读取文件并返回列表或字典

例如， `a.txt` 是文本文件，一共有5行内容。

```
a
b
c
d
e
```

#### 从硬盘读取

使用 `list_from_file` 读取 `a.txt`

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

同样， `b.txt` 也是文本文件，一共有3行内容

```
1 cat
2 dog cow
3 panda
```

使用 `dict_from_file` 读取 `b.txt`

```python
>>> mmcv.dict_from_file('b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

#### 从其他后端读取

使用 `list_from_file` 读取 `s3://bucket-name/a.txt`

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

使用 `dict_from_file` 读取 `b.txt`

```python
>>> mmcv.dict_from_file('s3://bucket-name/b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('s3://bucket-name/b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

### 读取和保存权重文件

#### 从硬盘读取权重文件或者将权重文件保存至硬盘

我们可以通过下面的方式从磁盘读取权重文件或者将权重文件保存至磁盘

```python
import torch

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = '/path/of/your/checkpoint2.pth'
# 从 filepath1 读取权重文件
checkpoint = torch.load(filepath1)
# 将权重文件保存至 filepath2
torch.save(checkpoint, filepath2)
```

MMCV 提供了很多后端，`HardDiskBackend` 是其中一个，我们可以通过它来读取或者保存权重文件。

```python
import io
from mmcv.fileio.file_client import HardDiskBackend

disk_backend = HardDiskBackend()
with io.BytesIO(disk_backend.get(filepath1)) as buffer:
    checkpoint = torch.load(buffer)
with io.BytesIO() as buffer:
    torch.save(checkpoint, f)
    disk_backend.put(f.getvalue(), filepath2)
```

如果我们想在接口中实现根据文件路径自动选择对应的后端，我们可以使用 `FileClient`。
例如，我们想实现两个方法，分别是读取权重以及保存权重，它们需支持不同类型的文件路径，可以是磁盘路径，也可以是网络路径或者其他路径。

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

#### 从网络远端读取权重文件

```{note}
目前只支持从网络远端读取权重文件，暂不支持将权重文件写入网络远端
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
