## 文件输入输出

文件输入输出模块提供了两个通用的 API 接口用于读取和保存不同格式的文件。

### 读取和保存数据

`mmcv` 提供了一个通用的 api 用于读取和保存数据，目前支持的格式有 json、yaml 和 pickle。

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

举 `PickleHandler` 为例。

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

使用 `list_from_file` 读取 `a.txt` 。

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

同样， `b.txt` 也是文本文件，一共有3行内容。

```
1 cat
2 dog cow
3 panda
```

使用 `dict_from_file` 读取 `b.txt` 。

```python
>>> mmcv.dict_from_file('b.txt')
{'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
>>> mmcv.dict_from_file('b.txt', key_type=int)
{1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```
