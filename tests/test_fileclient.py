import os.path as osp
import sys
import tempfile
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

        # test `get`
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

        # test `get_text`
        # input path is Path object
        value_buf = disk_backend.get_text(self.text_path)
        assert self.text_path.open('r').read() == value_buf
        # input path is str
        value_buf = disk_backend.get_text(str(self.text_path))
        assert self.text_path.open('r').read() == value_buf

        with tempfile.TemporaryDirectory() as tmp_dir:
            # test `put`
            filepath1 = Path(tmp_dir) / 'test.jpg'
            disk_backend.put(b'disk', filepath1)
            assert filepath1.open('rb').read() == b'disk'

            # test `put_text`
            filepath2 = Path(tmp_dir) / 'test.txt'
            disk_backend.put_text('disk', filepath2)
            assert filepath2.open('r').read() == 'disk'

            # test `isdir`
            assert disk_backend.isdir(tmp_dir)

            # test `isfile`
            assert disk_backend.isfile(filepath2)
            assert not disk_backend.isfile(Path(tmp_dir) / 'not/existed/path')

            # test `remove`
            disk_backend.remove(filepath2)

            # test `check_exist`
            assert not disk_backend.check_exist(filepath2)

            # test `copyfile`
            filepath3 = Path(tmp_dir) / 'test1.jpg'
            with pytest.raises(ValueError):
                # src_path and dst_path should both be a path of file rather
                # than a directory
                disk_backend.copyfile(src_path=tmp_dir, dst_path=tmp_dir)
            with pytest.raises(FileNotFoundError):
                # src_path should exist
                disk_backend.copyfile(
                    src_path='/not/existed/path', dst_path=filepath3)
            disk_backend.copyfile(filepath1, filepath3)
            assert disk_backend.isfile(filepath3)
            # force = False
            disk_backend.put(b'overwrite disk', filepath3)
            # filepath2 exists
            disk_backend.copyfile(filepath1, filepath3, force=False)
            assert filepath3.open('rb').read() == b'overwrite disk'
            # force = True
            disk_backend.copyfile(filepath1, filepath3, force=True)
            assert filepath3.open('rb').read() == b'disk'

        disk_dir = '/path/of/your/directory'
        assert disk_backend.concat_paths(disk_dir, 'file') == \
            osp.join(disk_dir, 'file')
        assert disk_backend.concat_paths(disk_dir, 'dir', 'file') == \
            osp.join(disk_dir, 'dir', 'file')

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
    @pytest.mark.parametrize('backend,prefix', [('petrel', None),
                                                (None, 's3')])
    def test_petrel_backend(self, backend, prefix):
        petrel_backend = FileClient(backend=backend, prefix=prefix)

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

        # test `_path_mapping`
        petrel_dir = 's3://user/data'
        petrel_backend = FileClient(
            'petrel', path_mapping={str(self.test_data_dir): petrel_dir})
        assert petrel_backend.client._path_mapping(str(self.img_path)) == \
            str(self.img_path).replace(str(self.test_data_dir), petrel_dir)

        petrel_path = f'{petrel_dir}/test.jpg'
        petrel_backend = FileClient('petrel')

        # test `_format_path`
        assert petrel_backend.client._format_path('s3://user\\data\\test.jpg')\
            == petrel_path

        # test `get`
        petrel_backend.client._client.Get = MagicMock(return_value=b'petrel')
        assert petrel_backend.get(petrel_path) == b'petrel'
        petrel_backend.client._client.Get.assert_called_with(petrel_path)

        # test `get_text`
        petrel_backend.client._client.Get = MagicMock(return_value=b'petrel')
        assert petrel_backend.get_text(petrel_path) == 'petrel'
        petrel_backend.client._client.Get.assert_called_with(petrel_path)

        # test `put`
        petrel_backend.client._client.put = MagicMock()
        petrel_backend.put(b'petrel', petrel_path)
        petrel_backend.client._client.put.assert_called_with(
            petrel_path, b'petrel')

        # test `put_text`
        petrel_backend.client._client.put = MagicMock()
        petrel_backend.put_text('petrel', petrel_path)
        petrel_backend.client._client.put.assert_called_with(
            petrel_path, b'petrel')

        # test `remove`
        petrel_backend.client._client.delete = MagicMock()
        petrel_backend.remove(petrel_path)
        petrel_backend.client._client.delete.assert_called_with(petrel_path)

        # test `check_exist`
        petrel_backend.client._client.contains = MagicMock(return_value=True)
        assert petrel_backend.check_exist(petrel_path)
        petrel_backend.client._client.contains.assert_called_with(petrel_path)

        # test `isdir`
        assert petrel_backend.isdir(f'{petrel_dir}/')
        # directory should end with '/'
        assert not petrel_backend.isdir(petrel_dir)

        # test `isfile`
        petrel_backend.client._client.contains = MagicMock(return_value=True)
        petrel_backend.client._client.contains = MagicMock(return_value=True)
        assert petrel_backend.isfile(petrel_path)
        petrel_backend.client._client.contains.assert_called_with(petrel_path)
        # if ending with '/', it is not a file
        assert not petrel_backend.isfile(f'{petrel_path}/')

        # test `concat_paths`
        assert petrel_backend.concat_paths(petrel_dir, 'file') == \
            f'{petrel_dir}/file'
        assert petrel_backend.concat_paths(petrel_dir, 'dir', 'file') == \
            f'{petrel_dir}/dir/file'

        # test `copyfile`
        with tempfile.TemporaryDirectory() as tmp_dir:
            disk_path = Path(tmp_dir) / 'test.jpg'
            with pytest.raises(ValueError):
                # src_path and dst_path should both be a path of file rather
                # than a directory
                petrel_backend.copyfile(src_path=petrel_dir, dst_path=tmp_dir)
            with pytest.raises(FileNotFoundError):
                # src_path should exist
                petrel_backend.client._client.contains = MagicMock(
                    return_value=False)
                petrel_backend.copyfile(
                    src_path=petrel_path, dst_path=disk_path)
            petrel_backend.client._client.contains = MagicMock(
                return_value=True)
            petrel_backend.copyfile(petrel_path, disk_path)
            assert osp.isfile(disk_path)
            assert disk_path.open('rb').read() == b'petrel'

            # force = False
            # filepath2 exists
            petrel_backend.client._client.Get = MagicMock(
                return_value=b'new petrel')
            petrel_backend.copyfile(petrel_path, disk_path, force=False)
            assert disk_path.open('rb').read() == b'petrel'
            # force = True
            petrel_backend.copyfile(petrel_path, disk_path, force=True)
            assert disk_path.open('rb').read() == b'new petrel'

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

    @pytest.mark.parametrize('backend,prefix', [('http', None),
                                                (None, 'http')])
    def test_http_backend(self, backend, prefix):
        http_backend = FileClient(backend=backend, prefix=prefix)
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

    def test_new_magic_method(self):

        class DummyBackend1(BaseStorageBackend):

            def get(self, filepath):
                return filepath

            def get_text(self, filepath, encoding='utf-8'):
                return filepath

        FileClient.register_backend('dummy_backend', DummyBackend1)
        client1 = FileClient(backend='dummy_backend')
        client2 = FileClient(backend='dummy_backend')
        assert client1 is client2

        # if a backend is overwrote, it will disable the singleton pattern for
        # the backend
        class DummyBackend2(BaseStorageBackend):

            def get(self, filepath):
                pass

            def get_text(self, filepath):
                pass

        FileClient.register_backend('dummy_backend', DummyBackend2, force=True)
        client3 = FileClient(backend='dummy_backend')
        client4 = FileClient(backend='dummy_backend')
        assert client3 is not client4

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
                return b'bytes2'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text2'

        # force=False
        with pytest.raises(KeyError):
            FileClient.register_backend('example', Example2Backend)

        FileClient.register_backend('example', Example2Backend, force=True)
        example_backend = FileClient('example')
        assert example_backend.get(self.img_path) == b'bytes2'
        assert example_backend.get_text(self.text_path) == 'text2'

        @FileClient.register_backend(name='example3')
        class Example3Backend(BaseStorageBackend):

            def get(self, filepath):
                return b'bytes3'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text3'

        example_backend = FileClient('example3')
        assert example_backend.get(self.img_path) == b'bytes3'
        assert example_backend.get_text(self.text_path) == 'text3'
        assert 'example3' in FileClient._backends

        # force=False
        with pytest.raises(KeyError):

            @FileClient.register_backend(name='example3')
            class Example4Backend(BaseStorageBackend):

                def get(self, filepath):
                    return b'bytes4'

                def get_text(self, filepath, encoding='utf-8'):
                    return 'text4'

        @FileClient.register_backend(name='example3', force=True)
        class Example5Backend(BaseStorageBackend):

            def get(self, filepath):
                return b'bytes5'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text5'

        example_backend = FileClient('example3')
        assert example_backend.get(self.img_path) == b'bytes5'
        assert example_backend.get_text(self.text_path) == 'text5'

        # prefixes is a str
        class Example6Backend(BaseStorageBackend):

            def get(self, filepath):
                return b'bytes6'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text6'

        FileClient.register_backend(
            'example4',
            Example6Backend,
            force=True,
            prefixes='example4_prefix')
        example_backend = FileClient('example4')
        assert example_backend.get(self.img_path) == b'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'
        example_backend = FileClient(prefix='example4_prefix')
        assert example_backend.get(self.img_path) == b'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'
        example_backend = FileClient('example4', prefix='example4_prefix')
        assert example_backend.get(self.img_path) == b'bytes6'
        assert example_backend.get_text(self.text_path) == 'text6'

        # prefixes is a list of str
        class Example7Backend(BaseStorageBackend):

            def get(self, filepath):
                return b'bytes7'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text7'

        FileClient.register_backend(
            'example5',
            Example7Backend,
            force=True,
            prefixes=['example5_prefix1', 'example5_prefix2'])
        example_backend = FileClient('example5')
        assert example_backend.get(self.img_path) == b'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'
        example_backend = FileClient(prefix='example5_prefix1')
        assert example_backend.get(self.img_path) == b'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'
        example_backend = FileClient(prefix='example5_prefix2')
        assert example_backend.get(self.img_path) == b'bytes7'
        assert example_backend.get_text(self.text_path) == 'text7'

        # backend has a higher priority than prefixes
        class Example8Backend(BaseStorageBackend):

            def get(self, filepath):
                return b'bytes8'

            def get_text(self, filepath, encoding='utf-8'):
                return 'text8'

        FileClient.register_backend(
            'example6',
            Example8Backend,
            force=True,
            prefixes='example6_prefix')
        example_backend = FileClient('example6')
        assert example_backend.get(self.img_path) == b'bytes8'
        assert example_backend.get_text(self.text_path) == 'text8'
        example_backend = FileClient('example6', prefix='example4_prefix')
        assert example_backend.get(self.img_path) == b'bytes8'
        assert example_backend.get_text(self.text_path) == 'text8'
