## 配置

`Config` 类用于操作配置文件，它支持从多种文件格式中加载配置，包括 **python**, **json** 和 **yaml**。
它提供了类似字典对象的接口来获取和设置值。

以配置文件 `test.py` 为例

```python
a = 1
b = dict(b1=[0, 1, 2], b2=None)
c = (1, 2)
d = 'string'
```

加载与使用配置文件

```python
>>> cfg = Config.fromfile('test.py')
>>> print(cfg)
>>> dict(a=1,
...      b=dict(b1=[0, 1, 2], b2=None),
...      c=(1, 2),
...      d='string')
```

对于所有格式的配置文件，都支持一些预定义变量。它会将 `{{ var }}` 替换为实际值。

目前支持以下四个预定义变量：

`{{ fileDirname }}` - 当前打开文件的目录名，例如 /home/your-username/your-project/folder

`{{ fileBasename }}` - 当前打开文件的文件名，例如 file.ext

`{{ fileBasenameNoExtension }}` - 当前打开文件不包含扩展名的文件名，例如 file

`{{ fileExtname }}` - 当前打开文件的扩展名，例如 .ext

这些变量名引用自 [VS Code](https://code.visualstudio.com/docs/editor/variables-reference)。

这里是一个带有预定义变量的配置文件的例子。

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

对于所有格式的配置文件, 都支持继承。为了重用其他配置文件的字段，
需要指定 `_base_='./config_a.py'` 或者一个包含配置文件的列表 `_base_=['./config_a.py', './config_b.py']`。

这里有 4 个配置继承关系的例子。

`config_a.py` 作为基类配置文件

```python
a = 1
b = dict(b1=[0, 1, 2], b2=None)
```

### 不含重复键值对从基类配置文件继承

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

在`config_b.py`里的新字段与在`config_a.py`里的旧字段拼接

### 含重复键值对从基类配置文件继承

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

在基类配置文件：`config_a` 里的 `b.b2=None`被配置文件：`config_c.py`里的 `b.b2=1`替代。

### 从具有忽略字段的配置文件继承

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

您还可以设置 `_delete_=True`忽略基类配置文件中的某些字段。所有在`b`中的旧键 `b1, b2, b3` 将会被新键 `b2, b3` 所取代。

### 从多个基类配置文件继承（基类配置文件不应包含相同的键）

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

### 从基类引用变量

您可以使用以下语法引用在基类中定义的变量。

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
