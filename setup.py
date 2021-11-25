import ctypes.util
import glob
import os
import os.path as osp
import platform
import re
import subprocess
import sys
from pathlib import Path
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

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
    # Modified from
    # https://github.com/pytorch/pytorch/blob/v1.3.1/tools/setup_helpers.
    IS_WINDOWS = (platform.system() == 'Windows')
    IS_DARWIN = (platform.system() == 'Darwin')
    IS_LINUX = (platform.system() == 'Linux')
    LINUX_HOME = '/usr/local/cuda'
    WINDOWS_HOME = glob.glob(
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')

    def check_env_flag(name, default=''):
        return os.getenv(name,
                         default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']

    def check_negative_env_flag(name, default=''):
        return os.getenv(name,
                         default).upper() in ['OFF', '0', 'NO', 'FALSE', 'N']

    def which(thefile):
        path = os.environ.get('PATH', os.defpath).split(os.pathsep)
        for d in path:
            fname = osp.join(d, thefile)
            fnames = [fname]
            if sys.platform == 'win32':
                exts = os.environ.get('PATHEXT', '').split(os.pathsep)
                fnames += [fname + ext for ext in exts]
            for name in fnames:
                if os.access(name, os.F_OK | os.X_OK) and not osp.isdir(name):
                    return name
        return None

    def find_nvcc():
        nvcc = which('nvcc')
        if nvcc is not None:
            return osp.dirname(nvcc)
        else:
            return None

    def find_cuda_version(cuda_home):
        if cuda_home is None:
            return None
        if IS_WINDOWS:
            candidate_names = [osp.basename(cuda_home)]
        else:
            # get CUDA lib folder
            cuda_lib_dirs = ['lib64', 'lib']
            for lib_dir in cuda_lib_dirs:
                cuda_lib_path = osp.join(cuda_home, lib_dir)
                if osp.exists(cuda_lib_path):
                    break
            # get a list of candidates for the version number
            # which are files containing cudart
            candidate_names = list(
                glob.glob(osp.join(cuda_lib_path, '*cudart*')))
            candidate_names = [osp.basename(c) for c in candidate_names]
            # if we didn't find any cudart, ask nvcc
            if len(candidate_names) == 0:
                proc = subprocess.Popen(['nvcc', '--version'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
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
            if not osp.exists(CUDA_HOME):
                # We use nvcc path on Linux and cudart path on macOS
                if IS_LINUX or IS_WINDOWS:
                    cuda_path = find_nvcc()
                else:
                    cudart_path = ctypes.util.find_library('cudart')
                    if cudart_path is not None:
                        cuda_path = osp.dirname(cudart_path)
                    else:
                        cuda_path = None
                if cuda_path is not None:
                    CUDA_HOME = osp.dirname(cuda_path)
                else:
                    CUDA_HOME = None
            cuda_version = find_cuda_version(CUDA_HOME)
        return cuda_version

    def get_gcc():
        proc = subprocess.Popen(['gcc', '--version'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate()
        gcc_version = re.search(r'[0-9]+\.[0-9]+\.[0-9]+', out.decode())
        if gcc_version:
            return gcc_version.group()
        else:
            return None

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

    mmcv_root = Path(__file__).parent
    version_path = mmcv_root / 'mmcv' / 'version.py'
    gcc_version = get_gcc()
    cuda_version = get_cuda()
    git_hash = get_git_hash(digits=7)
    if EXT_TYPE:
        torch_version = torch.__version__
        if '+' not in torch_version:  # eg. 1.8.1 -> 1.8.1+cu111
            if torch.cuda.is_available():
                torch_version += '+cu'
                torch_version += (torch.version.cuda).replace('.', '')
            else:
                torch_version += '+cpu'
    else:
        torch_version = None
    with open(version_path, 'r+') as f:
        lines = f.readlines()
        last_line = lines[-1]
        if '__all__' not in last_line:
            lines = lines[:-4]
        f.seek(0, 0)
        for line in lines:
            f.write(line)
        f.write(f"gcc_version = '{gcc_version}'\n"
                f"cuda_version = '{cuda_version}'\n"
                f"torch_version = '{torch_version}'\n"
                f"commit_id = '{git_hash}'\n")


# `pip install -e .` will run setup.py twice, first is
# `python setup.py egg_info`, second is `python setup.py develop`,
# we want to add version info only when `python setup.py develop`.
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
            osp.join(tensorrt_path, 'targets', '*', 'lib'))[0]
        library_dirs += [tensorrt_lib_path]
        libraries += ['nvinfer', 'nvparsers', 'nvinfer_plugin']
        libraries += ['cudart']
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = osp.abspath('./mmcv/ops/csrc/common/cuda')
        include_trt_path = osp.abspath('./mmcv/ops/csrc/tensorrt')
        include_dirs.append(include_path)
        include_dirs.append(include_trt_path)
        include_dirs.append(osp.join(tensorrt_path, 'include'))
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
        include_dirs.append(osp.abspath('./mmcv/ops/csrc/common'))
        include_dirs.append(osp.abspath('./mmcv/ops/csrc/common/cuda'))
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
            include_dirs.append(osp.abspath('./mmcv/ops/csrc/common/hip'))
        elif torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu')
            extension = CUDAExtension
            include_dirs.append(osp.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(osp.abspath('./mmcv/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp')
            extension = CppExtension
            include_dirs.append(osp.abspath('./mmcv/ops/csrc/common'))

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
        library_dirs += [osp.join(ort_path, 'lib')]
        libraries.append('onnxruntime')
        define_macros = []
        extra_compile_args = {'cxx': []}

        include_path = osp.abspath('./mmcv/ops/csrc/onnxruntime')
        include_dirs.append(include_path)
        include_dirs.append(osp.join(ort_path, 'include'))

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
