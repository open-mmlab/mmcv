# Copyright (c) Open-MMLab. All rights reserved.
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .io import dump, load, register_handler
from .parse import dict_from_file, list_from_file

__all__ = [
    'load', 'dump', 'register_handler', 'BaseFileHandler', 'JsonHandler',
    'PickleHandler', 'YamlHandler', 'list_from_file', 'dict_from_file'
]
