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
data = mmcv.load('test.pickle')
# load data from a file-like object
with open('test.json', 'r') as f:
    data = mmcv.load(f)

# dump data to a string
json_str = mmcv.dump(data, format='json')
# dump data to a file with a filename (infer format from file extension)
mmcv.dump(data, 'out.pickle')
# dump data to a file with a file-like object
with open('test.yaml', 'w') as f:
    data = mmcv.dump(data, f, format='yaml')
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
import mmcv

mmcv.list_from_file('a.txt')
# output ['a', 'b', 'c', 'd', 'e']
mmcv.list_from_file('a.txt', offset=2)
# output ['c', 'd', 'e']
mmcv.list_from_file('a.txt', max_num=2)
# output ['a', 'b']
mmcv.list_from_file('a.txt', prefix='/mnt/')
# output ['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

For example `b.txt` is a text file with 5 lines.
```
1 cat
2 dog cow
3 panda
```

Then use `dict_from_file` to load the list from a.txt.

```python
import mmcv

mmcv.dict_from_file('b.txt')
# output {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
mmcv.dict_from_file('b.txt', key_type=int)
# output {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```
