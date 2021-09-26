# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    # is_str_like_obj is a flag to mark which type of file object is processed,
    # bytes-like object or str-like object. For example, pickle only process
    # the bytes-like object and json only process the str-like object. The flag
    # will be used to check which type of buffer is used. If str-like object,
    # StringIO will be used. If bytes-like object, BytesIO will be used. The
    # usage of the flag can be found in `mmcv.load` or `mmcv.dump`
    is_str_like_obj = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode='r', **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
