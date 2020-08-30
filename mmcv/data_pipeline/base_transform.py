from abc import abstractmethod

from mmcv.utils import Registry


class BaseTransform:

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def required_keys(self):
        pass

    @property
    @abstractmethod
    def updated_keys(self):
        pass


TRANSFORMS = Registry('transforms')
