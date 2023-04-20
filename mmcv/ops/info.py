# Copyright (c) OpenMMLab. All rights reserved.
from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['get_compiler_version', 'get_compiling_cuda_version'])


def get_compiler_version():
    return ext_module.get_compiler_version()


def get_compiling_cuda_version():
    return ext_module.get_compiling_cuda_version()
