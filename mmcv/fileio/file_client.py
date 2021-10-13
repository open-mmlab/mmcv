# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os
import os.path as osp
import re
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlopen

from mmcv.utils.path import is_filepath


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
    """Ceph storage backend (for internal use).

    Args:
        path_mapping (dict|None): path mapping dict from local path to Petrel
            path. When ``path_mapping={'src': 'dst'}``, ``src`` in ``filepath``
            will be replaced by ``dst``. Default: None.

    .. warning::
        :class:`mmcv.fileio.file_client.CephBackend` will be deprecated,
        please use :class:`mmcv.fileio.file_client.PetrelBackend` instead.
    """

    def __init__(self, path_mapping=None):
        try:
            import ceph
        except ImportError:
            raise ImportError('Please install ceph to enable CephBackend.')

        warnings.warn(
            'CephBackend will be deprecated, please use PetrelBackend instead')
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

    PetrelBackend supports reading and writing data to multiple clusters.
    If the file path contains the cluster name, PetrelBackend will read data
    from specified cluster or write data to it. Otherwise, PetrelBackend will
    access the default cluster.

    Args:
        path_mapping (dict, optional): Path mapping dict from local path to
            Petrel path. When ``path_mapping={'src': 'dst'}``, ``src`` in
            ``filepath`` will be replaced by ``dst``. Default: None.
        enable_mc (bool, optional): Whether to enable memcached support.
            Default: True.

    Examples:
        >>> filepath1 = 's3://path/of/file'
        >>> filepath2 = 'cluster-name:s3://path/of/file'
        >>> client = PetrelBackend()
        >>> client.get(filepath1)  # get data from default cluster
        >>> client.get(filepath2)  # get data from 'cluster-name' cluster
    """

    def __init__(self,
                 path_mapping: Optional[dict] = None,
                 enable_mc: bool = True):
        try:
            from petrel_client import client
        except ImportError:
            raise ImportError('Please install petrel_client to enable '
                              'PetrelBackend.')

        self._client = client.Client(enable_mc=enable_mc)
        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping

    def _path_mapping(self, filepath: str) -> str:
        """Replace the prefix of ``filepath`` with :attr:`path_mapping`.

        Args:
            filepath (str): Path to be mapped.
        """
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of petrel oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a windows
        environment, the ``filepath`` will be the format of
        's3://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 's3://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def get(self, filepath: Union[str, Path]) -> memoryview:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.
        """
        filepath = self._path_mapping(str(filepath))
        filepath = self._format_path(filepath)
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str, optional): The encoding format used to open the
                ``filepath``. Default: 'utf-8'.
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.
        """
        filepath = self._path_mapping(str(filepath))
        filepath = self._format_path(filepath)
        self._client.put(filepath, obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Save data to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to encode the
                ``obj``. Default: 'utf-8'.
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        filepath = self._path_mapping(str(filepath))
        filepath = self._format_path(filepath)
        self._client.delete(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.
        """
        filepath = self._path_mapping(str(filepath))
        filepath = self._format_path(filepath)
        return self._client.contains(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether it is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.
        """
        return self.exists(filepath)

    def concat_paths(self, filepath: Union[str, Path],
                     *filepaths: Union[str, Path]) -> str:
        """Concatenate all filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.
        """
        formatted_paths = [
            self._format_path(self._path_mapping(str(filepath)))
        ]
        for path in filepaths:
            formatted_paths.append(
                self._format_path(self._path_mapping(str(path))))
        return '/'.join(formatted_paths)

    def _release_resource(self, filepath: str) -> None:
        """Release the resource generated by :meth:`_get_local_path`.

        Args:
            filepath (str): Path to be released.
        """
        os.remove(filepath)

    def _get_local_path(self, filepath: str) -> str:
        """Download a file from ``filepath``.

        Args:
            filepath (str): Download a file from ``filepath``.
        """
        assert self.isfile(filepath)

        # the file will be removed when calling _release_resource()
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(self.get(filepath))
        f.close()
        return f.name


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

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.
        """
        filepath = str(filepath)
        with open(filepath, 'rb') as f:
            value_buf = f.read()
        return value_buf

    def get_text(self,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str, optional): The encoding format used to open the
                ``filepath``. Default: 'utf-8'.
        """
        filepath = str(filepath)
        with open(filepath, 'r', encoding=encoding) as f:
            value_buf = f.read()
        return value_buf

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        filepath = str(filepath)
        with open(filepath, 'wb') as f:
            f.write(obj)

    def put_text(self,
                 obj: str,
                 filepath: Union[str, Path],
                 encoding: str = 'utf-8') -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Default: 'utf-8'.
        """
        filepath = str(filepath)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(obj)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.
        """
        filepath = str(filepath)
        os.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.
        """
        return osp.exists(str(filepath))

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether it is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.
        """
        return osp.isdir(str(filepath))

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether it is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.
        """
        return osp.isfile(str(filepath))

    def concat_paths(self, filepath: Union[str, Path],
                     *filepaths: Union[str, Path]) -> str:
        """Concatenate all filepaths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.
        """
        filepath = str(filepath)
        filepaths = [str(path) for path in filepaths]
        return osp.join(filepath, *filepaths)

    def _release_resource(self, filepath: str) -> None:
        """Do nothing in order to unify API."""
        pass

    def _get_local_path(self, filepath: str) -> str:
        """Do nothing in order to unify API."""
        return filepath


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get(self, filepath):
        value_buf = urlopen(filepath).read()
        return value_buf

    def get_text(self, filepath, encoding='utf-8'):
        value_buf = urlopen(filepath).read()
        return value_buf.decode(encoding)

    def _release_resource(self, filepath: str) -> None:
        """Release the resource generated by :meth:`_get_local_path`.

        Args:
            filepath (str): Path to be released.
        """
        os.remove(filepath)

    def _get_local_path(self, filepath: str) -> str:
        """Download a file from filepath.

        Args:
            filepath (str): Download a file from ``filepath``.
        """
        # the file will be removed when calling _release_resource()
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(self.get(filepath))
        f.close()
        return f.name


class FileClient:
    """A general file client to access files in different backends.

    The client loads a file or text in a specified backend from its path
    and returns it as a binary or text file. There are two ways to choose a
    backend, the name of backend and the prefix of path. Although both of them
    can be used to choose a storage backend, ``backend`` has a higher priority
    that is if they are all set, the storage backend will be chosen by the
    backend argument. If they are all `None`, the disk backend will be chosen.
    Note that It can also register other backend accessor with a given name,
    prefixes, and backend class. In addition, We use the singleton pattern to
    avoid repeated object creation. If the arguments are the same, the same
    object is returned.

    Args:
        backend (str, optional): The storage backend type. Options are "disk",
            "ceph", "memcached", "lmdb", "http" and "petrel". Default: None.
        prefix (str, optional): The prefix of the registered storage backend.
            Options are "s3", "http", "https". Default: None.

    Examples:
        >>> # only set backend
        >>> file_client = FileClient(backend='petrel')
        >>> # only set prefix
        >>> file_client = FileClient(prefix='s3')
        >>> # set both backend and prefix but use backend to choose client
        >>> file_client = FileClient(backend='petrel', prefix='s3')
        >>> # if the arguments are the same, the same object is returned
        >>> file_client1 = FileClient(backend='petrel')
        >>> file_client1 is file_client
        True

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
    # This collection is used to record the overridden backends, and when a
    # backend appears in the collection, the singleton pattern is disabled for
    # that backend, because if the singleton pattern is used, then the object
    # returned will be the backend before overwriting
    _overridden_backends = set()
    _prefix_to_backends = {
        's3': PetrelBackend,
        'http': HTTPBackend,
        'https': HTTPBackend,
    }
    _overridden_prefixes = set()

    _instances = {}

    def __new__(cls, backend=None, prefix=None, **kwargs):
        if backend is None and prefix is None:
            backend = 'disk'
        if backend is not None and backend not in cls._backends:
            raise ValueError(
                f'Backend {backend} is not supported. Currently supported ones'
                f' are {list(cls._backends.keys())}')
        if prefix is not None and prefix not in cls._prefix_to_backends:
            raise ValueError(
                f'prefix {prefix} is not supported. Currently supported ones '
                f'are {list(cls._prefix_to_backends.keys())}')

        # concatenate the arguments to a unique key for determining whether
        # objects with the same arguments were created
        arg_key = f'{backend}:{prefix}'
        for key, value in kwargs.items():
            arg_key += f':{key}:{value}'

        # if a backend was overridden, it will create a new object
        if (arg_key in cls._instances
                and backend not in cls._overridden_backends
                and prefix not in cls._overridden_prefixes):
            _instance = cls._instances[arg_key]
        else:
            # create a new object and put it to _instance
            _instance = super().__new__(cls)
            if backend is not None:
                _instance.client = cls._backends[backend](**kwargs)
                _instance.backend_name = backend
            else:
                _instance.client = cls._prefix_to_backends[prefix](**kwargs)
                # infer the backend name according to the prefix
                for backend_name, backend_cls in cls._backends.items():
                    if isinstance(_instance.client, backend_cls):
                        _instance.backend_name = backend_name
                        break
            cls._instances[arg_key] = _instance

        return _instance

    @staticmethod
    def parse_uri_prefix(uri: Union[str, Path]) -> Optional[str]:
        """Parse the prefix of a uri.

        Args:
            uri (str | Path): Uri to be parsed that contains the file prefix.

        Returns:
            return the prefix of uri if it contains "://" else None.

        Examples:
            >>> FileClient.parse_uri_prefix('s3://path/of/your/file')
            's3'
        """
        assert is_filepath(uri)
        uri = str(uri)
        if '://' not in uri:
            return None
        else:
            prefix, _ = uri.split('://')
            # In the case of PetrelBackend, the prefix may contains the cluster
            # name like clusterName:s3
            if ':' in prefix:
                _, prefix = prefix.split(':')
            return prefix

    @classmethod
    def infer_client(cls,
                     file_client_args: Optional[dict] = None,
                     uri: Optional[Union[str, Path]] = None) -> 'FileClient':
        """Infer a suitable file client based on the URI and arguments.

        Args:
            file_client_args (dict, optional): Arguments to instantiate a
                FileClient. Default: None.
            uri (str | Path, optional): Uri to be parsed that contains the file
                prefix. Default: None.

        Examples:
            >>> uri = 's3://path/of/your/file'
            >>> file_client = FileClient.infer_client(uri=uri)
            >>> file_client_args = {'backend': 'petrel'}
            >>> file_client = FileClient.infer_client(file_client_args)
        """
        assert file_client_args is not None or uri is not None
        if file_client_args is None:
            file_prefix = cls.parse_uri_prefix(uri)  # type: ignore
            return cls(prefix=file_prefix)
        else:
            return cls(**file_client_args)

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

        if name in cls._backends and force:
            cls._overridden_backends.add(name)
        cls._backends[name] = backend

        if prefixes is not None:
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            else:
                assert isinstance(prefixes, (list, tuple))
            for prefix in prefixes:
                if prefix not in cls._prefix_to_backends:
                    cls._prefix_to_backends[prefix] = backend
                elif (prefix in cls._prefix_to_backends) and force:
                    cls._overridden_prefixes.add(prefix)
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
            prefixes (str or list[str] or tuple[str], optional): The prefixes
                of the registered storage backend. Default: None.
                `New in version 1.3.15.`
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

    def get(self, filepath: Union[str, Path]) -> Union[bytes, memoryview]:
        """Read data from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.
        """
        return self.client.get(filepath)

    def get_text(self, filepath: Union[str, Path], encoding='utf-8') -> str:
        """Read data from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Default: 'utf-8'.
        """
        return self.client.get_text(filepath, encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'wb' mode.

        Args:
            obj (bytes): Data to be written.
            filepath (str or Path): Path to write data.
        """
        self.client.put(obj, filepath)

    def put_text(self, obj: str, filepath: Union[str, Path]) -> None:
        """Write data to a given ``filepath`` with 'w' mode.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str, optional): The encoding format used to open the
                `filepath`. Default: 'utf-8'.
        """
        self.client.put_text(obj, filepath)

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str, Path): Path to be removed.
        """
        self.client.remove(filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.
        """
        return self.client.exists(filepath)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether it is a directory.

        Args:
            filepath (str or Path): Path to be checked whether it is a
                directory.
        """
        return self.client.isdir(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check a ``filepath`` whether it is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.
        """
        return self.client.isfile(filepath)

    def concat_paths(self, filepath: Union[str, Path],
                     *filepaths: Union[str, Path]) -> str:
        """Concatenate all filepaths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.
        """
        return self.client.concat_paths(filepath, *filepaths)

    @contextmanager
    def get_local_path(self, filepath: Union[str, Path]):
        """Download data from ``filepath`` and write the data to local path.

        If the ``filepath`` is a local path, just return the itself.

        Note:
            ``get_local_path`` is an experimental interface that may change in
            the future.

        Args:
            filepath (str or Path): Path to be read data.

        Examples:
            >>> file_client = FileClient(prefix='s3')
            >>> with file_client.get_local_path('s3://bucket/abc.jpg') as path:
            ...     # do something here
        """
        path = self.client._get_local_path(str(filepath))
        try:
            yield path
        finally:
            self.client._release_resource(path)
