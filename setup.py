import platform
import re
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import Extension, dist, find_packages, setup

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])

import numpy  # NOQA: E402  # isort:skip
from Cython.Distutils import build_ext  # NOQA: E402  # isort:skip


def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary,
    else return secondary.
    """
    try:
        name = re.split(r'[!<>=]', primary)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(primary)


install_requires = ['addict', 'numpy', 'pyyaml']

# If first not installed install second package
CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3', 'opencv-python>=3')]

for main, secondary in CHOOSE_INSTALL_REQUIRES:
    install_requires.append(choose_requirement(main, secondary))


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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
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
