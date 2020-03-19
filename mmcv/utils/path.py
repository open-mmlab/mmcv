# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
from pathlib import Path

from .misc import is_str


def is_filepath(x):
    if is_str(x) or isinstance(x, Path):
        return True
    else:
        return False


def fopen(filepath, *args, **kwargs):
    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, recursive=False, fullpath=False):
    """Recursively scan a directory to find the interested files.

    Args:
        dir_path (str): Path of directory.
        suffix (str | tuple(str)): File suffix that we are interested in.
            Default: None.
        recursive (bool): If set to True, recursively scan the directory.
            Default: False.
        fullpath: (bool): If set to True, return the full pathes; otherwise,
            return file names. Default: False.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')
    with os.scandir(dir_path) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                if fullpath:
                    filename = entry.path
                else:
                    filename = entry.name
                if suffix is None:
                    yield filename
                elif filename.endswith(suffix):
                    yield filename
            else:
                if recursive:
                    yield from scandir(
                        entry,
                        suffix=suffix,
                        recursive=recursive,
                        fullpath=fullpath)
                else:
                    continue


def find_vcs_root(path, markers=('.git', )):
    """Finds the root directory (including itself) of specified markers.

    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.

    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None
