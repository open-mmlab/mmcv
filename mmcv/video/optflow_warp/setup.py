from distutils.core import setup, Extension
import numpy

from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules=[Extension("flow_warp_c",
                 sources=["flow_warp_module.pyx", "flow_warp.cpp"],
                 include_dirs=[numpy.get_include(), '.'],
                 language="c++")],
)