# Copyright (c) Open-MMLab. All rights reserved.
import json

import numpy as np

from .base import BaseFileHandler


def set_default(obj):
    """Set default json values for unserializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        json.dump(obj, file, default=set_default, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, default=set_default, **kwargs)
