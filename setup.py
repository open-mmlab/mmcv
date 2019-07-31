import platform
import sys
from io import open  # for Python 2 (identical to builtin in Python 3)
from setuptools import Extension, find_packages, setup, dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])

import numpy  # noqa: E402
from Cython.Distutils import build_ext  # noqa: E402

install_requires = [
    'numpy>=1.11.1', 'pyyaml', 'six', 'addict', 'requests', 'opencv-python',
    'Cython'
]
if sys.version_info < (3, 3):
    install_requires.append('backports.shutil_get_terminal_size')
if sys.version_info < (3, 4):
    install_requires.extend(['enum34', 'pathlib'])


def readme():
    with open('README.rst', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if platform.system() == 'Darwin':
    extra_compile_args = ['-stdlib=libc++']
    extra_link_args = ['-stdlib=libc++']
else:
    extra_compile_args = []
    extra_link_args = []

EXT_MODULES = [
    Extension(
        name='mmcv._ext',
        sources=[
            './mmcv/video/optflow_warp/flow_warp.cpp',
            './mmcv/video/optflow_warp/flow_warp_module.pyx'
        ],
        include_dirs=[numpy.get_include()],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='mmcv',
    version=get_version(),
    description='Open MMLab Computer Vision Foundation',
    long_description=readme(),
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='Kai Chen',
    author_email='chenkaidev@gmail.com',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=EXT_MODULES,
    cmdclass={'build_ext': build_ext},
    zip_safe=False)
