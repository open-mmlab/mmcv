import scipy.io as sio

from .base import BaseFileHandler


class MatHandler(BaseFileHandler):

    def load_from_path(self, filepath, **kwargs):
        return sio.loadmat(filepath, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        return sio.savemat(filepath, obj, **kwargs)
