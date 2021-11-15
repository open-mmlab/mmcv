import glob
import os
import re
import sys
from pathlib import Path
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

from mmcv.utils import get_cuda, get_gcc, get_git_hash

EXT_TYPE = ''
try:
    import torch
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import BuildExtension
        EXT_TYPE = 'parrots'
    else:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def add_version_info():
    mmcv_root = Path(__file__).parent
    version_path = mmcv_root / 'mmcv' / 'version.py'
    gcc_version = get_gcc()
    cuda_version = get_cuda()
    git_hash = get_git_hash()
    torch_version = torch.__version__
    with open(version_path, 'a') as f:
        f.write(f'\n'
                f"gcc_version = '{gcc_version}'\n"
                f"cuda_version = '{cuda_version}'\n"
                f"torch_version = '{torch_version}'\n"
                f"git_version = '{git_hash}'\n")


if 'egg_info' not in sys.argv:
    add_version_info()


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


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
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


install_requires = parse_requirements()

try:
    # OpenCV installed via conda.
    import cv2  # NOQA: F401
    major, minor, *rest = cv2.__version__.split('.')
    if int(major) < 3:
        raise RuntimeError(
            f'OpenCV >=3 is required but {cv2.__version__} is installed')
except ImportError:
    # If first not installed install second package
    CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
                                'opencv-python>=3')]
    for main, secondary in CHOOSE_INSTALL_REQUIRES:
        install_requires.append(choose_requirement(main, secondary))


def get_extensions():
    extensions = []

    if os.getenv('MMCV_WITH_TRT', '0') != '0':
        ext_name = 'mmcv._ext_trt'
        from torch.utils.cpp_extension import include_paths, library_paths
        library_dirs = []
        libraries = []
        include_dirs = []
        tensorrt_path = os.getenv('TENSORRT_DIR', '0')
        tensorrt_lib_path = glob.glob(
            os.path.join(tensorrt_path, 'targets', '*', 'lib'))[0]
        library_dirs += [tensorrt_lib_path]
        libraries += ['nvinfer', 'nvparsers', 'nvinfer_plugin']
        libraries += ['cudart']
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./mmcv/ops/csrc/common/cuda')
        include_trt_path = os.path.abspath('./mmcv/ops/csrc/tensorrt')
        include_dirs.append(include_path)
        include_dirs.append(include_trt_path)
        include_dirs.append(os.path.join(tensorrt_path, 'include'))
        include_dirs += include_paths(cuda=True)

        op_files = glob.glob('./mmcv/ops/csrc/tensorrt/plugins/*')
        define_macros += [('MMCV_WITH_CUDA', None)]
        define_macros += [('MMCV_WITH_TRT', None)]
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
        library_dirs += library_paths(cuda=True)

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
        extensions.append(ext_ops)

    if os.getenv('MMCV_WITH_OPS', '0') == '0':
        return extensions

    if EXT_TYPE == 'parrots':
        ext_name = 'mmcv._ext'
        from parrots.utils.build_extension import Extension
        # new parrots op impl do not use MMCV_USE_PARROTS
        # define_macros = [('MMCV_USE_PARROTS', None)]
        define_macros = []
        include_dirs = []
        op_files = glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') +\
            glob.glob('./mmcv/ops/csrc/parrots/*.cpp')
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args = {
            'nvcc': [cuda_args] if cuda_args else [],
            'cxx': [],
        }
        if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            extra_compile_args['nvcc'] += [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            cuda=True,
            pytorch=True)
        extensions.append(ext_ops)
    elif EXT_TYPE == 'pytorch':
        ext_name = 'mmcv._ext'
        from torch.utils.cpp_extension import CppExtension, CUDAExtension

        # prevent ninja from using too many resources
        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []
        extra_compile_args = {'cxx': []}
        include_dirs = []

        is_rocm_pytorch = False
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                       (ROCM_HOME is not None)) else False
        except ImportError:
            pass

        project_dir = 'mmcv/ops/csrc/'
        if is_rocm_pytorch:
            from torch.utils.hipify import hipify_python

            hipify_python.hipify(
                project_directory=project_dir,
                output_directory=project_dir,
                includes='mmcv/ops/csrc/*',
                show_detailed=True,
                is_pytorch_extension=True,
            )
            define_macros += [('MMCV_WITH_CUDA', None)]
            define_macros += [('HIP_DIFF', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/hip/*')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/hip'))
        elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

    if EXT_TYPE == 'pytorch' and os.getenv('MMCV_WITH_ORT', '0') != '0':
        ext_name = 'mmcv._ext_ort'
        from torch.utils.cpp_extension import library_paths, include_paths
        import onnxruntime
        library_dirs = []
        libraries = []
        include_dirs = []
        ort_path = os.getenv('ONNXRUNTIME_DIR', '0')
        library_dirs += [os.path.join(ort_path, 'lib')]
        libraries.append('onnxruntime')
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = os.path.abspath('./mmcv/ops/csrc/onnxruntime')
        include_dirs.append(include_path)
        include_dirs.append(os.path.join(ort_path, 'include'))

        op_files = glob.glob('./mmcv/ops/csrc/onnxruntime/cpu/*')
        if onnxruntime.get_device() == 'GPU' or os.getenv('FORCE_CUDA',
                                                          '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files += glob.glob('./mmcv/ops/csrc/onnxruntime/gpu/*')
            include_dirs += include_paths(cuda=True)
            library_dirs += library_paths(cuda=True)
        else:
            include_dirs += include_paths(cuda=False)
            library_dirs += library_paths(cuda=False)

        from setuptools import Extension
        ext_ops = Extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            language='c++',
            library_dirs=library_dirs,
            libraries=libraries)
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
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Contributors',
    author_email='openmmlab@gmail.com',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
