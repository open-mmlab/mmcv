import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mmcv
from mmcv import BaseStorageBackend, FileClient

sys.modules['ceph'] = MagicMock()
sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()
sys.modules['mc'] = MagicMock()


class MockS3Client:

    def __init__(self, enable_mc=True):
        self.enable_mc = enable_mc

    def Get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content


class MockPetrelClient:

    def __init__(self, enable_mc=True, enable_multi_cluster=False):
        self.enable_mc = enable_mc
        self.enable_multi_cluster = enable_multi_cluster

    def Get(self, filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        return content


class MockMemcachedClient:

    def __init__(self, server_list_cfg, client_cfg):
        pass

    def Get(self, filepath, buffer):
        with open(filepath, 'rb') as f:
            buffer.content = f.read()


class TestFileClient:

    @classmethod
    def setup_class(cls):
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.img_shape = (300, 400, 3)
        cls.text_path = cls.test_data_dir / 'filelist.txt'

    def test_error(self):
        with pytest.raises(ValueError):
            FileClient('hadoop')

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

        # `path_mapping` is either None or dict
        with pytest.raises(AssertionError):
            FileClient('ceph', path_mapping=1)
        # test `path_mapping`
        ceph_path = 's3://user/data'
        ceph_backend = FileClient(
            'ceph', path_mapping={str(self.test_data_dir): ceph_path})
        ceph_backend.client._client.Get = MagicMock(
            return_value=ceph_backend.client._client.Get(self.img_path))
        img_bytes = ceph_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape
        ceph_backend.client._client.Get.assert_called_with(
            str(self.img_path).replace(str(self.test_data_dir), ceph_path))

    @patch('petrel_client.client.Client', MockPetrelClient)
    @pytest.mark.parametrize('backend,prefixes', [('petrel', None),
                                                  (None, 's3')])
    def test_petrel_backend(self, backend, prefixes):
        petrel_backend = FileClient(backend=backend, prefixes=prefixes)

        # input path is Path object
        img_bytes = petrel_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape
        # input path is str
        img_bytes = petrel_backend.get(str(self.img_path))
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape

        # `path_mapping` is either None or dict
        with pytest.raises(AssertionError):
            FileClient('petrel', path_mapping=1)
        # test `path_mapping`
        petrel_path = 's3://user/data'
        petrel_backend = FileClient(
            'petrel', path_mapping={str(self.test_data_dir): petrel_path})
        petrel_backend.client._client.Get = MagicMock(
            return_value=petrel_backend.client._client.Get(self.img_path))
        img_bytes = petrel_backend.get(self.img_path)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape
        petrel_backend.client._client.Get.assert_called_with(
            str(self.img_path).replace(str(self.test_data_dir), petrel_path))
        # remove a file
        petrel_backend.client._client.delete = MagicMock()
        petrel_backend.remove(self.img_path)
        petrel_backend.client._client.delete.assert_called_with(
            str(self.img_path).replace(str(self.test_data_dir), petrel_path))

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

    @pytest.mark.parametrize('backend,prefixes', [('http', None),
                                                  (None, 'http')])
    def test_http_backend(self, backend, prefixes):
        http_backend = FileClient(backend=backend, prefixes=prefixes)
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmcv/' \
            'master/tests/data/color.jpg'
        text_url = 'https://raw.githubusercontent.com/open-mmlab/mmcv/' \
            'master/tests/data/filelist.txt'

        # input is path or Path object
        with pytest.raises(Exception):
            http_backend.get(self.img_path)
        with pytest.raises(Exception):
            http_backend.get(str(self.img_path))
        with pytest.raises(Exception):
            http_backend.get_text(self.text_path)
        with pytest.raises(Exception):
            http_backend.get_text(str(self.text_path))

        # input url is http image
        img_bytes = http_backend.get(img_url)
        img = mmcv.imfrombytes(img_bytes)
        assert img.shape == self.img_shape

        # input url is http text
        value_buf = http_backend.get_text(text_url)
        assert self.text_path.open('r').read() == value_buf

    def test_parse_uri_prefix(self):
        # input path is None
        with pytest.raises(AssertionError):
            FileClient.parse_uri_prefix(None)
        # input path is list
        with pytest.raises(AssertionError):
            FileClient.parse_uri_prefix([])

        # input path is Path object
        assert FileClient.parse_uri_prefix(self.img_path) is None
        # input path is str
        assert FileClient.parse_uri_prefix(str(self.img_path)) is None

        # input path starts with https
        img_url = 'https://raw.githubusercontent.com/open-mmlab/mmcv/' \
            'master/tests/data/color.jpg'
        assert FileClient.parse_uri_prefix(img_url) == 'https'

        # input path starts with s3
        img_url = 's3://your_bucket/img.png'
        assert FileClient.parse_uri_prefix(img_url) == 's3'

        # input path starts with clusterName:s3
        img_url = 'clusterName:s3://your_bucket/img.png'
        assert FileClient.parse_uri_prefix(img_url) == 's3'

    def test_infer_client(self):
        # HardDiskBackend
        file_client_args = {'backend': 'disk'}
        client = FileClient.infer_client(file_client_args)
        assert client.backend_name == 'disk'
        client = FileClient.infer_client(uri=self.img_path)
        assert client.backend_name == 'disk'

        # PetrelBackend
        file_client_args = {'backend': 'petrel'}
        client = FileClient.infer_client(file_client_args)
        assert client.backend_name == 'petrel'
        uri = 's3://user_data'
        client = FileClient.infer_client(uri=uri)
        assert client.backend_name == 'petrel'

    def test_register_backend(self):

        # name must be a string
        with pytest.raises(TypeError):

            class TestClass1:
                pass

            FileClient.register_backend(1, TestClass1)

        # module must be a class
        with pytest.raises(TypeError):
            FileClient.register_backend('int', 0)

        # module must be a subclass of BaseStorageBackend
        with pytest.raises(TypeError):

            class TestClass1:
                pass

            FileClient.register_backend('TestClass1', TestClass1)

        class ExampleBackend(BaseStorageBackend):

            def get(self, filepath):
                return filepath

            def get_text(self, filepath, encoding='utf-8'):
                return filepath

        FileClient.register_backend('example', ExampleBackend)
        example_backend = FileClient('example')
        assert example_backend.get(self.img_path) == self.img_path
        assert example_backend.get_text(self.text_path) == self.text_path
        assert 'example' in FileClient._backends

        class Example2Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes2'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text2'

        # force=False
        with pytest.raises(KeyError):
            FileClient.register_backend('example', Example2Backend)

        FileClient.register_backend('example', Example2Backend, force=True)
        example_backend = FileClient('example')
        assert example_backend.get(self.img_path) == 'bytes2'
        assert example_backend.get_text(self.text_path) == 'text2'

        @FileClient.register_backend(name='example3')
        class Example3Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes3'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text3'

        example_backend = FileClient('example3')
        assert example_backend.get(self.img_path) == 'bytes3'
        assert example_backend.get_text(self.text_path) == 'text3'
        assert 'example3' in FileClient._backends

        # force=False
        with pytest.raises(KeyError):

            @FileClient.register_backend(name='example3')
            class Example4Backend(BaseStorageBackend):

                def get(self, filepath):
                    return 'bytes4'

                def get_text(self, filepath, encoding='utf-8'):
                    return 'text4'

        @FileClient.register_backend(name='example3', force=True)
        class Example5Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes5'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text5'

        example_backend = FileClient('example3')
        assert example_backend.get(self.img_path) == 'bytes5'
        assert example_backend.get_text(self.text_path) == 'text5'

        # prefixes is a str
        class Example6Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes6'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text6'

        FileClient.register_backend(
            'example4',
            Example6Backend,
            force=True,
            prefixes='example4_prefix')
        example_backend = FileClient('example4')
        assert example_backend.get(self.img_path) == 'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'
        example_backend = FileClient(prefixes='example4_prefix')
        assert example_backend.get(self.img_path) == 'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'
        example_backend = FileClient('example4', prefixes='example4_prefix')
        assert example_backend.get(self.img_path) == 'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'

        # prefixes is a list of str
        class Example7Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes7'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text7'

        FileClient.register_backend(
            'example5',
            Example7Backend,
            force=True,
            prefixes=['example5_prefix1', 'example5_prefix2'])
        example_backend = FileClient('example5')
        assert example_backend.get(self.img_path) == 'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'
        example_backend = FileClient(prefixes='example5_prefix1')
        assert example_backend.get(self.img_path) == 'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'
        example_backend = FileClient(prefixes='example5_prefix2')
        assert example_backend.get(self.img_path) == 'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'

        # backend has a higher priority than prefixes
        class Example8Backend(BaseStorageBackend):

            def get(self, filepath):
                return 'bytes8'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text8'

        FileClient.register_backend(
            'example6',
            Example8Backend,
            force=True,
            prefixes='example6_prefix')
        example_backend = FileClient('example6')
        assert example_backend.get(self.img_path) == 'bytes8'
        assert example_backend.get_text(self.text_path) == 'text8'
        example_backend = FileClient('example6', prefixes='example4_prefix')
        assert example_backend.get(self.img_path) == 'bytes8'
        assert example_backend.get_text(self.text_path) == 'text8'
