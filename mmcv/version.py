# Copyright (c) Open-MMLab. All rights reserved.
from .utils import parse_version_info

__version__ = '1.1.1'

version_info = parse_version_info(__version__)

__all__ = ['__version__', 'version_info']
