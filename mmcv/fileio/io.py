from .processors import JsonProcessor, PickleProcessor, YamlProcessor
from ..utils import is_str

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
    if file_format is None and is_str(file):
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
