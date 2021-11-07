from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

setup(
    ext_modules=cythonize(Extension(
            "*", ['*.pyx'], extra_compile_args=["/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c",
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
        )
    ),
    include_dirs=[numpy.get_include()],

    )


