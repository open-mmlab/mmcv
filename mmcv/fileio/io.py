# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO, StringIO
from pathlib import Path

from ..utils import is_list_of, is_str
from .file_client import FileClient
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}


def load(file, file_format=None, file_client_args=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is None:
        file_prefix = FileClient.parse_uri_prefix(file)
        client = FileClient(prefixes=file_prefix)
    else:
        client = FileClient(**file_client_args)

    handler = file_handlers[file_format]
    if is_str(file):
        if handler.str_like_obj:
            with StringIO(client.get_text(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            with BytesIO(client.get(file)) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, file_client_args=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dump to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    if file_client_args is None:
        file_prefix = FileClient.parse_uri_prefix(file)
        client = FileClient(prefixes=file_prefix)
    else:
        client = FileClient(**file_client_args)

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if handler.str_like_obj:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                client.put_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                client.put(f.getvalue(), file)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler, file_formats):
    """Register a handler for some file extensions.

    Args:
        handler (:obj:`BaseFileHandler`): Handler to be registered.
        file_formats (str or list[str]): File formats to be handled by this
            handler.
    """
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(
            f'handler must be a child of BaseFileHandler, not {type(handler)}')
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError('file_formats must be a str or a list of str')
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):

    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
