import glob
import os
import re
import setuptools
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import dist, find_packages, setup

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])

import numpy  # NOQA: E402  # isort:skip
from Cython.Build import cythonize  # NOQA: E402  # isort:skip

EXT_TYPE = ''
try:
    import torch
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import BuildExtension
        EXT_TYPE = 'parrots'
    else:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
except ModuleNotFoundError:
    from Cython.Distutils import build_ext as BuildExtension
    print('Skip building ext ops due to the absence of torch.')


def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary, else
    return secondary."""
    try:
        name = re.split(r'[!<>=]', primary)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(primary)


def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# If first not installed install second package
CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3', 'opencv-python>=3')]

install_requires = parse_requirements()
for main, secondary in CHOOSE_INSTALL_REQUIRES:
    install_requires.append(choose_requirement(main, secondary))


def get_extensions():
    extensions = []

    ext_flow = setuptools.Extension(
        name='mmcv._flow_warp_ext',
        sources=[
            './mmcv/video/optflow_warp/flow_warp.cpp',
            './mmcv/video/optflow_warp/flow_warp_module.pyx'
        ],
        include_dirs=[numpy.get_include()],
        language='c++')
    extensions.extend(cythonize(ext_flow))

    if os.getenv('MMCV_WITH_OPS', '0') == '0':
        return extensions

    if EXT_TYPE == 'parrots':
        ext_name = 'mmcv._ext'
        from parrots.utils.build_extension import Extension
        define_macros = [('MMCV_USE_PARROTS', None)]
        op_files = glob.glob('./mmcv/ops/csrc/parrots/*')
        include_path = os.path.abspath('./mmcv/ops/csrc')
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=[include_path],
            define_macros=define_macros,
            extra_compile_args={
                'nvcc': [cuda_args] if cuda_args else [],
                'cxx': [],
            },
            cuda=True)
        extensions.append(ext_ops)
    elif EXT_TYPE == 'pytorch':
        ext_name = 'mmcv._ext'
        from torch.utils.cpp_extension import (CUDAExtension, CppExtension)
        # prevent ninja from using too many resources
        os.environ.setdefault('MAX_JOBS', '4')
        define_macros = []
        extra_compile_args = {'cxx': []}

        if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*')
            extension = CUDAExtension
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp')
            extension = CppExtension

        include_path = os.path.abspath('./mmcv/ops/csrc')
        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=[include_path],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)
    return extensions


setup(
    name='mmcv' if os.getenv('MMCV_WITH_OPS', '0') == '0' else 'mmcv-full',
    version=get_version(),
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Authors',
    author_email='openmmlab@gmail.com',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False)
