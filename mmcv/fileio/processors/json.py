import json

from .base import BaseFileProcessor


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
