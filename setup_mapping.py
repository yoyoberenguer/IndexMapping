# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    name='mapping',
    ext_modules=cythonize(Extension(
            "*", ['*.pyx'], extra_compile_args=["/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c",
        )
    ),
    include_dirs=[numpy.get_include()],

    )


