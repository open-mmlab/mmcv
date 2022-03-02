# Copyright (c) OpenMMLab. All rights reserved.
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock


@contextmanager
def mock_package(*package_name):
    try:
        for name in package_name:
            sys.modules[name] = MagicMock()
        yield
    finally:
        for name in package_name:
            del sys.modules[name]
