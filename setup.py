import sys
from setuptools import find_packages, setup

install_requires = [
    'numpy>=1.11.1', 'pyyaml', 'six', 'addict', 'requests', 'opencv-python'
]
if sys.version_info < (3, 3):
    install_requires.append('backports.shutil_get_terminal_size')
if sys.version_info < (3, 4):
    install_requires.extend(['enum34', 'pathlib'])


def readme():
    with open('README.rst') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


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
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='Kai Chen',
    author_email='chenkaidev@gmail.com',
    license='GPLv3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    zip_safe=False)
