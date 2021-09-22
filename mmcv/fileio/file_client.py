# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlopen


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: ``get()`` and ``get_text()``.
    ``get()`` reads the file as a byte stream and ``get_text()`` reads the file
    as texts.
    """

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass


class CephBackend(BaseStorageBackend):
    """Ceph storage backend.

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When ``path_mapping={'src': 'dst'}``, ``src`` in ``filepath``
            will be replaced by ``dst``. Default: None.
    """

    def __init__(self, path_mapping=None):
        try:
            import ceph
        except ImportError:
            raise ImportError('Please install ceph to enable CephBackend.')

        self._client = ceph.S3Client()
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def get(self, filepath):
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class PetrelBackend(BaseStorageBackend):
    """Petrel storage backend (for internal use).

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When `path_mapping={'src': 'dst'}`, `src` in `filepath` will
            be replaced by `dst`. Default: None.
        enable_mc (bool): whether to enable memcached support. Default: True.
        enable_multi_cluster (bool): Whether to enable multiple clusters.
            Default: False.
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True,
                 enable_multi_cluster: bool = False):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(
            enable_mc=enable_mc, enable_multi_cluster=enable_multi_cluster)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def get(self, filepath: Union[str, Path]) -> memoryview:
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        self._client.put(filepath, obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        self.put(bytes(obj, encoding=encoding), str(filepath))

    def remove(self, filepath: Union[str, Path]) -> None:
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        self._client.delete(filepath)


class MemcachedBackend(BaseStorageBackend):
    """Memcached storage backend.

    Attributes:
        server_list_cfg (str): Config file for memcached server list.
        client_cfg (str): Config file for memcached client.
        sys_path (str | None): Additional path to be appended to `sys.path`.
            Default: None.
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)
        try:
            import mc
        except ImportError:
            raise ImportError(
                'Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg,
                                                      self.client_cfg)
        # mc.pyvector servers as a point which points to a memory cache
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        filepath = str(filepath)
        import mc
        self._client.Get(filepath, self._mc_buffer)
        value_buf = mc.ConvertBuffer(self._mc_buffer)
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class LmdbBackend(BaseStorageBackend):
    """Lmdb storage backend.

    Args:
        db_path (str): Lmdb database path.
        readonly (bool, optional): Lmdb environment parameter. If True,
            disallow any write operations. Default: True.
        lock (bool, optional): Lmdb environment parameter. If False, when
            concurrent access occurs, do not lock the database. Default: False.
        readahead (bool, optional): Lmdb environment parameter. If False,
            disable the OS filesystem readahead mechanism, which may improve
            random read performance when a database is larger than RAM.
            Default: False.

    Attributes:
        db_path (str): Lmdb database path.
    """

    def __init__(self,
                 db_path,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 **kwargs):
        try:
            import lmdb
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        self.db_path = str(db_path)
        self._client = lmdb.open(
            self.db_path,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            **kwargs)

    def get(self, filepath):
        """Get values according to the filepath.

        Args:
            filepath (str | obj:`Path`): Here, filepath is the lmdb key.
        """
        filepath = str(filepath)
        with self._client.begin(write=False) as txn:
            value_buf = txn.get(filepath.encode('ascii'))
        return value_buf

    def get_text(self, filepath, encoding=None):
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """Raw hard disks storage backend."""

    def get(self, filepath):
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self, filepath, encoding='utf-8'):
        filepath = str(filepath)
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj, filepath):
        filepath = str(filepath)
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self, obj, filepath, encoding='utf-8'):
        filepath = str(filepath)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath):
        value_buf = urlopen(filepath).read()
        return value_buf

    def get_text(self, filepath, encoding='utf-8'):
        value_buf = urlopen(filepath).read()
        return value_buf.decode(encoding)


class FileClient:
    """A general file client to access files in different backend.

    The client loads a file or text in a specified backend from its path
    and return it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefixes of path. Although both of
    them can be used to choose a storage backend, ``backend`` has a higher
    priority that is if they are all set, the storage backend will be chosen by
    the backend argument. If they are all `None`, the disk backend will be
    chosen. Note that It can also register other backend accessor with a given
    name, prefixes, and backend class.

    Args:
        backend (str): The storage backend type. Options are "disk", "ceph",
            "memcached", "lmdb", "http" and "petrel". Default: None.
        prefixes (str or list[str] or tuple[str]): The prefixes of the
            registered storage backend. Options are "s3", "http", "https".
            Default: None.

    .. versionadd:: 1.3.14
            The *prefixes* parameter.

    Example:
        >>> # only set backend
        >>> file_client = FileClient(backend='ceph')
        >>> # only set prefixes
        >>> file_client = FileClient(prefixes='s3')
        >>> # set both backend and prefixes but use backend to choose client
        >>> file_client = FileClient(backend='ceph', prefixes='s3')

    Attributes:
        client (:obj:`BaseStorageBackend`): The backend object.
    """

    _backends = {
        'disk': HardDiskBackend,
        'ceph': CephBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
        'petrel': PetrelBackend,
        'http': HTTPBackend,
    }
    _prefix_to_backends = {
        's3': PetrelBackend,
        'http': HTTPBackend,
        'https': HTTPBackend,
    }

    def __init__(self, backend=None, prefixes=None, **kwargs):
        if backend is None and prefixes is None:
            backend = 'disk'
        if backend is not None and backend not in self._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(self._backends.keys())}')
        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))

            if not set(prefixes).issubset(self._prefix_to_backends.keys()):
                raise ValueError(
                    f'prefixes {prefixes} is not supported. Currently '
                    'supported ones are '
                    f'{list(self._prefix_to_backends.keys())}')

        if backend is not None:
            self.client = self._backends[backend](**kwargs)
        else:
            for prefix in prefixes:
                self.client = self._prefix_to_backends[prefix](**kwargs)
                break

        for name, backend_cls in self._backends.items():
            if isinstance(self.client, backend_cls):
                self.backend_name = name
                break

    @staticmethod
    def parse_uri_prefix(uri):
        uri = str(uri)
        if '://' not in uri:
            return None
        else:
            prefix, _ = uri.split('://')
            # In the case of ceph, the prefix may contains the cluster name
            # like clusterName:s3
            if ':' in prefix:
                _, prefix = prefix.split(':')
            return prefix

    @classmethod
    def _register_backend(cls, name, backend, force=False, prefixes=None):
        if not isinstance(name, str):
            raise TypeError('the backend name should be a string, '
                            f'but got {type(name)}')
        if not inspect.isclass(backend):
            raise TypeError(
                f'backend should be a class but got {type(backend)}')
        if not issubclass(backend, BaseStorageBackend):
            raise TypeError(
                f'backend {backend} is not a subclass of BaseStorageBackend')
        if not force and name in cls._backends:
            raise KeyError(
                f'{name} is already registered as a storage backend, '
                'add "force=True" if you want to override it')

        cls._backends[name] = backend
        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if (prefix not in cls._prefix_to_backends) or force:
                    cls._prefix_to_backends[prefix] = backend
                else:
                    raise KeyError(
                        f'{prefix} is already registered as a storage backend,'
                        ' add "force=True" if you want to override it')

    @classmethod
    def register_backend(cls, name, backend=None, force=False, prefixes=None):
        """Register a backend to FileClient.

        This method can be used as a normal class method or a decorator.

        .. code-block:: python

            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

            FileClient.register_backend('new', NewBackend)

        or

        .. code-block:: python

            @FileClient.register_backend('new')
            class NewBackend(BaseStorageBackend):

                def get(self, filepath):
                    return filepath

                def get_text(self, filepath):
                    return filepath

        Args:
            name (str): The name of the registered backend.
            backend (class, optional): The backend class to be registered,
                which must be a subclass of :class:`BaseStorageBackend`.
                When this method is used as a decorator, backend is None.
                Defaults to None.
            force (bool, optional): Whether to override the backend if the name
                has already been registered. Defaults to False.
            prefixes (str or list[str] or tuple[str]): The prefix of the
                registered storage backend.

        .. versionadd:: 1.3.14
            The *prefixes* parameter.
        """
        if backend is not None:
            cls._register_backend(
                name, backend, force=force, prefixes=prefixes)
            return

        def _register(backend_cls):
            cls._register_backend(
                name, backend_cls, force=force, prefixes=prefixes)
            return backend_cls

        return _register

    def get(self, filepath):
        return self.client.get(filepath)

    def get_text(self, filepath, encoding='utf-8'):
        return self.client.get_text(filepath, encoding)

    def put(self, obj, filepath):
        self.client.put(obj, filepath)

    def put_text(self, obj, filepath):
        self.client.put_text(obj, filepath)

    def remove(self, filepath):
        self.client.remove(filepath)
