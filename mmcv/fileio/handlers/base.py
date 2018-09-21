from abc import ABCMeta, abstractmethod


class BaseFileHandler(object):

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
