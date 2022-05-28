"""CLI to manage extension module"""
import os
import os.path as osp
import tempfile
import urllib.request
import zipfile
from setuptools.extern.packaging.tags import sys_tags
from typing import Tuple

import click

import mmcv

# from pip._internal.utils.compatibility_tags import get_supported

MMCV_FULL_FIND_URL = ('https://download.openmmlab.com/mmcv/dist/'
                      '{cuda_version}/{torch_version}/index.html')

MMCV_FULL_WHL_URL = ('https://download.openmmlab.com/mmcv/dist/'
                     '{cuda_version}/{torch_version}/'
                     'mmcv_full-{version}-{tag}.whl')

MMCV_FULL_WHL_URL_NAME = ('https://download.openmmlab.com/mmcv/dist/'
                          '{cuda_version}/{torch_version}/{name}')


# Port from mim
def get_torch_cuda_version() -> Tuple[str, str]:
    """Get PyTorch version and CUDA version if it is available.

    Example:
        >>> get_torch_cuda_version()
        '1.8.0', '102'
    """
    try:
        import torch
    except ImportError as err:
        raise err

    torch_v = torch.__version__
    if '+' in torch_v:  # 1.8.1+cu111 -> 1.8.1
        torch_v = torch_v.split('+')[0]

    if torch.version.cuda is not None:
        # torch.version.cuda like 10.2 -> 102
        cuda_v = ''.join(torch.version.cuda.split('.'))
    else:
        cuda_v = 'cpu'
    return torch_v, cuda_v


# Port from mim
def infer_url(name: str, version: str) -> str:
    """Try to infer find_url if possible.

    If package is the official package, the find_url can be inferred.

    Args:
        package (str): The name of package, such as mmcls.
    """
    torch_v, cuda_v = get_torch_cuda_version()

    # In order to avoid builiding mmcv-full from source, we ignore the
    # difference among micro version because there are usually no big
    # changes among micro version. For example, the mmcv-full built in
    # pytorch 1.8.0 also works on 1.8.1 or other versions.
    major, minor, *_ = torch_v.split('.')
    torch_v = '.'.join([major, minor, '0'])

    if cuda_v.isdigit():
        cuda_v = f'cu{cuda_v}'

    if version is None:
        version = mmcv.version.__version__

    tag = next(sys_tags())

    if name is None:
        whl_url = MMCV_FULL_WHL_URL.format(
            version=version,
            cuda_version=cuda_v,
            torch_version=f'torch{torch_v}',
            tag=str(tag))
    else:
        whl_url = MMCV_FULL_WHL_URL_NAME.format(
            version=version, cuda_version=cuda_v, name=name)

    find_url = MMCV_FULL_FIND_URL.format(
        cuda_version=cuda_v, torch_version=f'torch{torch_v}')

    return find_url, whl_url


def _list():
    root_dir = osp.dirname(mmcv.__file__)
    exts = []
    for o in os.listdir(root_dir):
        if o.startswith('_ext') and osp.isfile(osp.join(root_dir, o)):
            exts.append(o)
    return root_dir, exts


def _install(url, name, version):
    if url is None:
        find_url, url = infer_url(name, version)
        base_name = url.split('/')[-1]
        download_path = osp.join(tempfile.gettempdir(), base_name)

    if osp.exists(download_path):
        print(f"use cache {download_path}.")
    else:
        print(f"Downloading {url} to {download_path}.")

    try:
        urllib.request.urlretrieve(url, download_path)
    except urllib.error.HTTPError as e:
        raise Exception(f'Download failed from {url}.\n'
                        f'Please check {find_url} for avalable wheels'
                        f'and re-try mmcv init [--version VERSION] with'
                        f'an available version.') from e

    file = zipfile.ZipFile(download_path)

    for n in file.namelist():
        if n.startswith('mmcv/_ext.'):
            import mmcv
            ext_dir = osp.dirname(osp.dirname(mmcv.__file__))
            print(f"Extracting {n} to {ext_dir}")
            file.extract(n, ext_dir)
            break

    try:
        import mmcv._ext
    except ImportError as e:
        raise Exception(f'Installed failed.') from e

    print(f"Install successed.")


@click.command()
@click.option(
    "-u",
    "--url",
    default=None,
    required=False,
    help='URL to the desired mmcv-full wheel.')
@click.option(
    "-n",
    "--name",
    default=None,
    required=False,
    help='Name of the desired mmcv-full wheel.')
@click.option(
    "-v",
    "--version",
    default=None,
    required=False,
    help='Version of the desired mmcv-full.')
def install(url, name, version):
    '''Initialize extension module from pre-compiled mmcv-full.

    Examples:

    1. Find the suitable wheel to download, infer all versions and platforms automatically:

        mmcv ext install

    2. Install extension module based on MMCV version, infer other informations automatically.

        mmcv ext init -v 1.5.0

    3. Install extension module based on wheel name, infer find-url automatically.

        mmcv ext init -n mmcv_full-1.5.0-cp39-cp39-win_amd64.whl
        mmcv ext init -n mmcv_full-1.5.0-cp39-cp39-win_amd64

        Search give n name under https://download.openmmlab.com/mmcv/dist/{CUDA_VERSION}/{PYTORCH_VERSION}/indexl.html

    4. Install extension module from given download URL:

        mmcv ext init -u https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/mmcv_full-1.5.0-cp39-cp39-win_amd64.whl
    '''
    _install(url, name, version)


@click.command()
def list():
    '''List all installed extension modules.'''
    root_dir, exts = _list()
    if exts:
        click.echo(f"Extension module installed under {root_dir}:\n" +
                   '\n'.join(exts))
    else:
        click.echo(f"No extension module installed under {root_dir}.")


@click.command()
def clean():
    '''Clean all installed extension modules.'''
    root_dir, exts = _list()
    if not exts:
        click.echo(f"No extension module installed under {root_dir}.")
    else:
        if click.confirm(f"Remove all extension modules under {root_dir}?\n" +
                         '\n'.join(exts) + '\n'):
            for ext in exts:
                os.remove(osp.join(root_dir, ext))


@click.group()
def ext():
    """Managing extension modules."""
    pass


ext.add_command(install)
ext.add_command(list)
ext.add_command(clean)
