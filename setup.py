# BUILD THE PROJECT WITH THE CORRECT PYTHON VERSION
# pip install wheel

# python setup.py bdist_wheel
# OR python setup.py sdist bdist_wheel (to include the source)

# for python 3.8
# C:\Users\yoann\AppData\Roaming\Python\Python38\Scripts\twine upload
# --verbose --repository testpypi dist/IndexMapping-1.0.3-cp38-cp38-win_amd64.whl

# for python 3.6
# C:\Users\yoann\AppData\Roaming\Python\Python36\Scripts\twine upload
# --verbose --repository testpypi dist/IndexMapping-1.0.3-cp36-cp36-win_amd64.whl

# python setup.py bdist_wheel
# twine upload --verbose --repository testpypi dist/*

# PRODUCTION v:
# version 1.0.2
# C:\Users\yoann\AppData\Roaming\Python\Python38\Scripts\twine upload --verbose dist/IndexMapping-1.0.2*

# CREATING EXECUTABLE
# pyinstaller --onefile pyinstaller_config.spec

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

import setuptools
from Cython.Build import cythonize
from setuptools import setup, Extension

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name                         ="IndexMapping",
    version                      ="1.0.1",
    author                       ="Yoann Berenguer",
    author_email                 ="yoyoberenguer@hotmail.com",
    description                  ="1d array transpose/conversion",
    long_description             =long_description,
    long_description_content_type="text/markdown",
    url                          ="https://github.com/yoyoberenguer/IndexMapping",
    packages                     =setuptools.find_packages(),
    ext_modules                  =cythonize([
        Extension("IndexMapping.mapping", ["mapping.pyx"],
                  extra_compile_args=["/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c"),
        Extension("IndexMapping.mapcfunctions", ["mapcfunctions.pyx"],
                  extra_compile_args=["/Qpar", "/fp:fast", "/O2", "/Oy", "/Ot"], language="c")]),
    include_dirs=[numpy.get_include()],
    license                      ='MIT',

    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Cython',
        'Programming Language :: C',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        # 'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        # 'Programming Language :: Python :: 3 :: Only',
    ],

    install_requires=[
        'setuptools>=49.2.1',
        'Cython>=0.28'
    ],
    python_requires         ='>=3.0',
    platforms               =['any'],
    include_package_data    =True,
    data_files=[('./lib/site-packages/IndexMapping',
                 ['__init__.pxd',
                  '__init__.py',
                  'pyproject.toml',
                  'setup_mapping.py',
                  'mapcfunctions.pyx',
                  'mapping.pxd',
                  'mapping.pyx',
                  'LICENSE',
                  'README.md',
                  'requirements.txt'

                  ]),
                ('./lib/site-packages/IndexMapping/test',
                 [
                  'test/test_mapping.py',
                  'test/test_split.py',
                  'test/profiling.py'
                 ]),

                ('./lib/site-packages/IndexMapping/Assets',
                 [
                  'Assets/A1.png',
                 ])
                ],

    project_urls = {  # Optional
                   'Bug Reports': 'https://github.com/yoyoberenguer/IndexMapping/issues',
                   'Source'     : 'https://github.com/yoyoberenguer/IndexMapping',
               },
)
