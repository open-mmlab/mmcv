# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from
https://github.com/pytorch/pytorch/blob/v1.3.1/tools/setup_helpers."""
import ctypes.util
import glob
import os
import platform
import re
import subprocess
import sys
import warnings
from subprocess import PIPE, Popen

from packaging.version import parse

IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
LINUX_HOME = '/usr/local/cuda'
WINDOWS_HOME = glob.glob(
    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')


def check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_negative_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']


def which(thefile):
    path = os.environ.get('PATH', os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def find_nvcc():
    nvcc = which('nvcc')
    if nvcc is not None:
        return os.path.dirname(nvcc)
    else:
        return None


def find_cuda_version(cuda_home):
    if cuda_home is None:
        return None
    if IS_WINDOWS:
        candidate_names = [os.path.basename(cuda_home)]
    else:
        # get CUDA lib folder
        cuda_lib_dirs = ['lib64', 'lib']
        for lib_dir in cuda_lib_dirs:
            cuda_lib_path = os.path.join(cuda_home, lib_dir)
            if os.path.exists(cuda_lib_path):
                break
        # get a list of candidates for the version number
        # which are files containing cudart
        candidate_names = list(
            glob.glob(os.path.join(cuda_lib_path, '*cudart*')))
        candidate_names = [os.path.basename(c) for c in candidate_names]
        # if we didn't find any cudart, ask nvcc
        if len(candidate_names) == 0:
            proc = Popen(['nvcc', '--version'], stdout=PIPE, stderr=PIPE)
            out, err = proc.communicate()
            candidate_names = [out.decode().rsplit('V')[-1]]

    # suppose version is MAJOR.MINOR.PATCH, all numbers
    version_regex = re.compile(r'[0-9]+\.[0-9]+\.[0-9]+')
    candidates = [
        c.group() for c in map(version_regex.search, candidate_names) if c
    ]
    if len(candidates) > 0:
        # normally only one will be retrieved, take the first result
        return candidates[0]
    # if no candidates were found, try MAJOR.MINOR
    version_regex = re.compile(r'[0-9]+\.[0-9]+')
    candidates = [
        c.group() for c in map(version_regex.search, candidate_names) if c
    ]
    if len(candidates) > 0:
        return candidates[0]
    return None


def get_cuda():
    if check_negative_env_flag('USE_CUDA') or check_env_flag('USE_ROCM'):
        cuda_version = None
    else:
        if IS_LINUX or IS_DARWIN:
            CUDA_HOME = os.getenv('CUDA_HOME', LINUX_HOME)
        else:
            CUDA_HOME = os.getenv('CUDA_PATH', '').replace('\\', '/')
            if CUDA_HOME == '' and len(WINDOWS_HOME) > 0:
                CUDA_HOME = WINDOWS_HOME[0].replace('\\', '/')
        if not os.path.exists(CUDA_HOME):
            # We use nvcc path on Linux and cudart path on macOS
            if IS_LINUX or IS_WINDOWS:
                cuda_path = find_nvcc()
            else:
                cudart_path = ctypes.util.find_library('cudart')
                if cudart_path is not None:
                    cuda_path = os.path.dirname(cudart_path)
                else:
                    cuda_path = None
            if cuda_path is not None:
                CUDA_HOME = os.path.dirname(cuda_path)
            else:
                CUDA_HOME = None
        cuda_version = find_cuda_version(CUDA_HOME)
    return cuda_version


def get_gcc():
    proc = Popen(['gcc', '--version'], stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    gcc_version = re.search(r'[0-9]+\.[0-9]+\.[0-9]+', out.decode())
    if gcc_version:
        return gcc_version.group()
    else:
        return None


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_git_hash(fallback='unknown', digits=None):
    """Get the git hash of the current repo.

    Args:
        fallback (str, optional): The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits (int, optional): kept digits of the hash. Defaults to None,
            meaning all digits are kept.

    Returns:
        str: Git commit hash.
    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback

    return sha
