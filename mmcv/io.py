from abc import ABCMeta, abstractmethod
import json

import yaml
from six.moves import cPickle as pickle
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from mmcv.utils import is_str

__all__ = ['load', 'dump', 'list_from_file', 'dict_from_file']


class BaseFileProcessor(object):

    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def load_from_path(filepath, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def load_from_fileobj(file, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def dump_to_str(obj, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def dump_to_path(obj, filepath, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def dump_to_fileobj(obj, file, **kwargs):
        pass


class JsonProcessor(BaseFileProcessor):

    @staticmethod
    def load_from_path(filepath):
        with open(filepath, 'r') as f:
            obj = json.load(f)
        return obj

    @staticmethod
    def load_from_fileobj(file):
        return json.load(file)

    @staticmethod
    def dump_to_str(obj, **kwargs):
        return json.dumps(obj, **kwargs)

    @staticmethod
    def dump_to_path(obj, filepath, **kwargs):
        with open(filepath, 'w') as f:
            json.dump(obj, f, **kwargs)

    @staticmethod
    def dump_to_fileobj(obj, file, **kwargs):
        json.dump(obj, file, **kwargs)


class YamlProcessor(BaseFileProcessor):

    @staticmethod
    def load_from_path(filepath, **kwargs):
        kwargs.setdefault('Loader', Loader)
        with open(filepath, 'r') as f:
            obj = yaml.load(f, **kwargs)
        return obj

    @staticmethod
    def load_from_fileobj(file, **kwargs):
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    @staticmethod
    def dump_to_str(obj, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)

    @staticmethod
    def dump_to_path(obj, filepath, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        with open(filepath, 'w') as f:
            yaml.dump(obj, f, **kwargs)

    @staticmethod
    def dump_to_fileobj(obj, file, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)


class PickleProcessor(BaseFileProcessor):

    @staticmethod
    def load_from_path(filepath, **kwargs):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f, **kwargs)
        return obj

    @staticmethod
    def load_from_fileobj(file, **kwargs):
        return pickle.load(file, **kwargs)

    @staticmethod
    def dump_to_str(obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    @staticmethod
    def dump_to_path(obj, filepath, **kwargs):
        kwargs.setdefault('protocol', 2)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f, **kwargs)

    @staticmethod
    def dump_to_fileobj(obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)


file_processors = {
    'json': JsonProcessor,
    'yaml': YamlProcessor,
    'yml': YamlProcessor,
    'pickle': PickleProcessor,
    'pkl': PickleProcessor
}


def load(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or file-like object): Filename or a file-like object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".

    Returns:
        The content from the file.
    """
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in file_processors:
        raise TypeError('Unsupported format: {}'.format(file_format))

    processor = file_processors[file_format]
    if is_str(file):
        obj = processor.load_from_path(file, **kwargs)
    elif hasattr(file, 'read'):
        obj = processor.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or file-like object, optional): If not specified, then the
            object is dump to a str, otherwise to a file specified by the
            filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Returns:
        bool: True for success, False otherwise
    """
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_processors:
        raise TypeError('Unsupported format: {}'.format(file_format))

    processor = file_processors[file_format]
    if file is None:
        return processor.dump_to_str(obj, **kwargs)
    elif is_str(file):
        processor.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, 'write'):
        processor.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def list_from_file(filename, prefix='', offset=0, max_num=0):
    """Load a text file and parse the content as a list of strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the begining of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip('\n'))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns splited by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict's keys. str is user by default and
            type conversion will be performed if specified.

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    with open(filename, 'r') as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
