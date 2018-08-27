from six.moves import cPickle as pickle

from .base import BaseFileProcessor


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
    