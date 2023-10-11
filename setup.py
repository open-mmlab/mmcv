import glob
import os
import platform
import re
import warnings
from pkg_resources import DistributionNotFound, get_distribution, parse_version
from setuptools import find_packages, setup

EXT_TYPE = ''
try:
    import torch
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import BuildExtension
        EXT_TYPE = 'parrots'
    elif (hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()) or \
            os.getenv('FORCE_MLU', '0') == '1':
        from torch_mlu.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    else:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
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
    with open(version_file, encoding='utf-8') as f:
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
        with open(fpath) as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    yield from parse_line(line)

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

        # Following strings of text style are from colorama package
        bright_style, reset_style = '\x1b[1m', '\x1b[0m'
        red_text, blue_text = '\x1b[31m', '\x1b[34m'
        white_background = '\x1b[107m'

        msg = white_background + bright_style + red_text
        msg += 'DeprecationWarning: ' + \
            'Custom TensorRT Ops will be deprecated in future. '
        msg += blue_text + \
            'Welcome to use the unified model deployment toolbox '
        msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
        msg += reset_style
        warnings.warn(msg)

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
        # prevent cub/thrust conflict with other python library
        # More context See issues #1454
        extra_compile_args['nvcc'] += ['-Xcompiler=-fno-gnu-unique']
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
            glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') +\
            glob.glob('./mmcv/ops/csrc/parrots/*.cpp')
        op_files.remove('./mmcv/ops/csrc/pytorch/cuda/iou3d_cuda.cu')
        op_files.remove('./mmcv/ops/csrc/pytorch/cpu/bbox_overlaps_cpu.cpp')
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
        include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        cuda_args = os.getenv('MMCV_CUDA_ARGS')
        extra_compile_args = {
            'nvcc': [cuda_args, '-std=c++14'] if cuda_args else ['-std=c++14'],
            'cxx': ['-std=c++14'],
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

        # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
        # required key passed to PyTorch. Even if there is no flag passed
        # to cxx, users also need to pass an empty list to PyTorch.
        # Since PyTorch1.8.0, it has a default value so users do not need
        # to pass an empty list anymore.
        # More details at https://github.com/pytorch/pytorch/pull/45956
        extra_compile_args = {'cxx': []}

        # Since the PR (https://github.com/open-mmlab/mmcv/pull/1463) uses
        # c++14 features, the argument ['std=c++14'] must be added here.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['-std=c++14']
            else:
                extra_compile_args['cxx'] = ['-std=c++17']
        else:
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['/std:c++14']
            else:
                extra_compile_args['cxx'] = ['/std:c++17']

        include_dirs = []

        extra_objects = []
        extra_link_args = []
        is_rocm_pytorch = False
        try:
            from torch.utils.cpp_extension import ROCM_HOME
            is_rocm_pytorch = True if ((torch.version.hip is not None) and
                                       (ROCM_HOME is not None)) else False
        except ImportError:
            pass

        if is_rocm_pytorch or torch.cuda.is_available() or os.getenv(
                'FORCE_CUDA', '0') == '1':
            if is_rocm_pytorch:
                define_macros += [('MMCV_WITH_HIP', None)]
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            if is_rocm_pytorch and platform.system() != 'Windows':
                extra_compile_args['nvcc'] += \
                    ['--gpu-max-threads-per-block=1024']
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/pytorch'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        elif (hasattr(torch, 'is_mlu_available') and
                torch.is_mlu_available()) or \
                os.getenv('FORCE_MLU', '0') == '1':
            from torch_mlu.utils.cpp_extension import MLUExtension

            def get_mluops_version(file_path):
                with open(file_path) as f:
                    for line in f:
                        if re.search('MLUOP_MAJOR', line):
                            major = line.strip().split(' ')[2]
                        if re.search('MLUOP_MINOR', line):
                            minor = line.strip().split(' ')[2]
                        if re.search('MLUOP_PATCHLEVEL', line):
                            patchlevel = line.strip().split(' ')[2]
                mluops_version = f'v{major}.{minor}.{patchlevel}'
                return mluops_version

            mmcv_mluops_version = get_mluops_version(
                './mmcv/ops/csrc/pytorch/mlu/mlu_common_helper.h')
            mlu_ops_path = os.getenv('MMCV_MLU_OPS_PATH')
            if mlu_ops_path:
                exists_mluops_version = get_mluops_version(
                    mlu_ops_path + '/bangc-ops/mlu_op.h')
                if exists_mluops_version != mmcv_mluops_version:
                    print('the version of mlu-ops provided is %s,'
                          ' while %s is needed.' %
                          (exists_mluops_version, mmcv_mluops_version))
                    exit()
                try:
                    if os.path.exists('mlu-ops'):
                        if os.path.islink('mlu-ops'):
                            os.remove('mlu-ops')
                            os.symlink(mlu_ops_path, 'mlu-ops')
                        elif os.path.abspath('mlu-ops') != mlu_ops_path:
                            os.symlink(mlu_ops_path, 'mlu-ops')
                    else:
                        os.symlink(mlu_ops_path, 'mlu-ops')
                except Exception:
                    raise FileExistsError(
                        'mlu-ops already exists, please move it out,'
                        'or rename or remove it.')
            else:
                if not os.path.exists('mlu-ops'):
                    import requests
                    mluops_url = 'https://github.com/Cambricon/mlu-ops/' + \
                        'archive/refs/tags/' + mmcv_mluops_version + '.zip'
                    req = requests.get(mluops_url)
                    with open('./mlu-ops.zip', 'wb') as f:
                        try:
                            f.write(req.content)
                        except Exception:
                            raise ImportError('failed to download mlu-ops')

                    from zipfile import BadZipFile, ZipFile
                    with ZipFile('./mlu-ops.zip', 'r') as archive:
                        try:
                            archive.extractall()
                            dir_name = archive.namelist()[0].split('/')[0]
                            os.rename(dir_name, 'mlu-ops')
                        except BadZipFile:
                            print('invalid mlu-ops.zip file')
                else:
                    exists_mluops_version = get_mluops_version(
                        './mlu-ops/bangc-ops/mlu_op.h')
                    if exists_mluops_version != mmcv_mluops_version:
                        print('the version of provided mlu-ops is %s,'
                              ' while %s is needed.' %
                              (exists_mluops_version, mmcv_mluops_version))
                        exit()

            define_macros += [('MMCV_WITH_MLU', None)]
            mlu_args = os.getenv('MMCV_MLU_ARGS', '-DNDEBUG ')
            mluops_includes = []
            mluops_includes.append('-I' +
                                   os.path.abspath('./mlu-ops/bangc-ops'))
            mluops_includes.append(
                '-I' + os.path.abspath('./mlu-ops/bangc-ops/kernels'))
            extra_compile_args['cncc'] = [mlu_args] + \
                mluops_includes if mlu_args else mluops_includes
            extra_compile_args['cxx'] += ['-fno-gnu-unique']
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/mlu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/common/mlu/*.mlu') + \
                glob.glob(
                    './mlu-ops/bangc-ops/core/**/*.cpp', recursive=True) + \
                glob.glob(
                    './mlu-ops/bangc-ops/kernels/**/*.cpp', recursive=True) + \
                glob.glob(
                    './mlu-ops/bangc-ops/kernels/**/*.mlu', recursive=True)
            extra_link_args = [
                '-Wl,--whole-archive',
                './mlu-ops/bangc-ops/kernels/kernel_wrapper/lib/libextops.a',
                '-Wl,--no-whole-archive'
            ]
            extension = MLUExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/mlu'))
            include_dirs.append(os.path.abspath('./mlu-ops/bangc-ops'))
        elif (hasattr(torch.backends, 'mps')
              and torch.backends.mps.is_available()) or os.getenv(
                  'FORCE_MPS', '0') == '1':
            # objc compiler support
            from distutils.unixccompiler import UnixCCompiler
            if '.mm' not in UnixCCompiler.src_extensions:
                UnixCCompiler.src_extensions.append('.mm')
                UnixCCompiler.language_map['.mm'] = 'objc'

            define_macros += [('MMCV_WITH_MPS', None)]
            extra_compile_args = {}
            extra_compile_args['cxx'] = ['-Wall', '-std=c++17']
            extra_compile_args['cxx'] += [
                '-framework', 'Metal', '-framework', 'Foundation'
            ]
            extra_compile_args['cxx'] += ['-ObjC++']
            # src
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')
            # TODO: support mps ops on torch>=2.1.0
            if parse_version(torch.__version__) < parse_version('2.1.0'):
                op_files += glob.glob('./mmcv/ops/csrc/common/mps/*.mm') + \
                    glob.glob('./mmcv/ops/csrc/pytorch/mps/*.mm')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/mps'))
        elif (os.getenv('FORCE_NPU', '0') == '1'):
            print(f'Compiling {ext_name} only with CPU and NPU')
            try:
                from torch_npu.utils.cpp_extension import NpuExtension
                define_macros += [('MMCV_WITH_NPU', None)]
                extension = NpuExtension
                if parse_version(torch.__version__) <= parse_version('2.0.0'):
                    define_macros += [('MMCV_WITH_XLA', None)]
                if parse_version(torch.__version__) > parse_version('2.0.0'):
                    define_macros += [('MMCV_WITH_KPRIVATE', None)]
            except Exception:
                raise ImportError('can not find any torch_npu')
            # src
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/common/npu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/npu/*.cpp')
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/npu'))
        else:
            print(f'Compiling {ext_name} only with CPU')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))

        # Since the PR (https://github.com/open-mmlab/mmcv/pull/1463) uses
        # c++14 features, the argument ['std=c++14'] must be added here.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_objects=extra_objects,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args)
        extensions.append(ext_ops)

    if EXT_TYPE == 'pytorch' and os.getenv('MMCV_WITH_ORT', '0') != '0':

        # Following strings of text style are from colorama package
        bright_style, reset_style = '\x1b[1m', '\x1b[0m'
        red_text, blue_text = '\x1b[31m', '\x1b[34m'
        white_background = '\x1b[107m'

        msg = white_background + bright_style + red_text
        msg += 'DeprecationWarning: ' + \
            'Custom ONNXRuntime Ops will be deprecated in future. '
        msg += blue_text + \
            'Welcome to use the unified model deployment toolbox '
        msg += 'MMDeploy: https://github.com/open-mmlab/mmdeploy'
        msg += reset_style
        warnings.warn(msg)
        ext_name = 'mmcv._ext_ort'
        import onnxruntime
        from torch.utils.cpp_extension import include_paths, library_paths
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
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Contributors',
    author_email='openmmlab@gmail.com',
    install_requires=install_requires,
    extras_require={
        'all': parse_requirements('requirements.txt'),
        'tests': parse_requirements('requirements/test.txt'),
        'build': parse_requirements('requirements/build.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
    },
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
