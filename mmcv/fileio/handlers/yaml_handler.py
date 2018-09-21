import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from .base import BaseFileHandler


class YamlHandler(BaseFileHandler):

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
