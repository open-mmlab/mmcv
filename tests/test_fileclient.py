import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mmcv
from mmcv import BaseStorageBackend, FileClient

sys.modules['ceph'] = MagicMock()
sys.modules['mc'] = MagicMock()


class MockS3Client(object):

    def Get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content


class MockMemcachedClient(object):

    def __init__(self, server_list_cfg, client_cfg):
        pass

    def Get(self, filepath, buffer):
        with open(filepath, 'rb') as f:
            buffer.content = f.read()


class TestFileClient(object):

    @classmethod
    def setup_class(cls):
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.img_shape = (300, 400, 3)
        cls.text_path = cls.test_data_dir / 'filelist.txt'

    def test_disk_backend(self):
        disk_backend = FileClient('disk')

        # input path is Path object
        img_bytes = disk_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert self.img_path.open('rb').read() == img_bytes
        assert img.shape == self.img_shape
        # input path is str
        img_bytes = disk_backend.get(str(self.img_path))
        img = mmcv.imfrombytes(img_bytes)
        assert self.img_path.open('rb').read() == img_bytes
        assert img.shape == self.img_shape

        # input path is Path object
        value_buf = disk_backend.get_text(self.text_path)
        assert self.text_path.open('r').read() == value_buf
        # input path is str
        value_buf = disk_backend.get_text(str(self.text_path))
        assert self.text_path.open('r').read() == value_buf

    @patch('ceph.S3Client', MockS3Client)
    def test_ceph_backend(self):
        ceph_backend = FileClient('ceph')

        # input path is Path object
        with pytest.raises(NotImplementedError):
            ceph_backend.get_text(self.text_path)
        # input path is str
        with pytest.raises(NotImplementedError):
            ceph_backend.get_text(str(self.text_path))

        # input path is Path object
        img_bytes = ceph_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape
        # input path is str
        img_bytes = ceph_backend.get(str(self.img_path))
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape

    @patch('mc.MemcachedClient.GetInstance', MockMemcachedClient)
    @patch('mc.pyvector', MagicMock)
    @patch('mc.ConvertBuffer', lambda x: x.content)
    def test_memcached_backend(self):
        mc_cfg = dict(server_list_cfg='', client_cfg='', sys_path=None)
        mc_backend = FileClient('memcached', **mc_cfg)

        # input path is Path object
        with pytest.raises(NotImplementedError):
            mc_backend.get_text(self.text_path)
        # input path is str
        with pytest.raises(NotImplementedError):
            mc_backend.get_text(str(self.text_path))

        # input path is Path object
        img_bytes = mc_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape
        # input path is str
        img_bytes = mc_backend.get(str(self.img_path))
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape

    def test_lmdb_backend(self):
        lmdb_path = self.test_data_dir / 'demo.lmdb'

        # db_path is Path object
        lmdb_backend = FileClient('lmdb', db_path=lmdb_path)

        with pytest.raises(NotImplementedError):
            lmdb_backend.get_text(self.text_path)

        img_bytes = lmdb_backend.get('baboon')
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == (120, 125, 3)

        # db_path is str
        lmdb_backend = FileClient('lmdb', db_path=str(lmdb_path))
        with pytest.raises(NotImplementedError):
            lmdb_backend.get_text(str(self.text_path))
        img_bytes = lmdb_backend.get('baboon')
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == (120, 125, 3)

    def test_register_backend(self):
        with pytest.raises(TypeError):

            class TestClass1(object):
                pass

            FileClient.register_backend('TestClass1', TestClass1)

        with pytest.raises(TypeError):
            FileClient.register_backend('int', 0)

        class ExampleBackend(BaseStorageBackend):

            def get(self, filepath):
                return filepath

            def get_text(self, filepath):
                return filepath

        FileClient.register_backend('example', ExampleBackend)
        example_backend = FileClient('example')
        assert example_backend.get(self.img_path) == self.img_path
        assert example_backend.get_text(self.text_path) == self.text_path
        assert 'example' in FileClient._backends

    def test_error(self):
        with pytest.raises(ValueError):
            FileClient('hadoop')
