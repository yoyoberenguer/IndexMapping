# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True
# encoding: utf-8


## License :
"""
```
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
"""
# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")


# CYTHON IS REQUIRED
try:
    cimport cython
    from cython.parallel cimport prange
except ImportError:
    raise ImportError("\n<cython> library is missing on your system."
          "\nTry: \n   C:\\pip install cython on a window command prompt.")

from libc.stdio cimport printf
cimport numpy as np


__version__ = "1.0.2"

"""
__version__ = 1.0.1 to 1.0.2 
+   Removed warning during compilation signed/unsigned mismatch warning (C4018 in Visual Studio).
    Replacing int with some unsigned type is a problem because we frequently use OpenMP pragmas, 
    and it requires the counter to be int.
+   Added define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] to setup.py file and setup_mapping.py to 
    get rid of this warning
+   corrected the variables x, y, z (now unsigned int) in mapping.pxd  
    cdef struct xyz:
        unsigned int x;
        unsigned int y;
        unsigned int z;
+   Changed vfb_c function call to inline, e.g
    cdef inline unsigned char [::1] vfb_c(unsigned char [:] source,
                unsigned char [::1] target, int width, int height)nogil:
+  Added 
   assert width > 0, 'Argument width cannot be <=0'
   assert height > 0, 'Argument height cannot be <=0'
   in functions vfb_rgb & vfb_rgba
   
+  Corrected the __init__.py and __init__.pxd 
+  Changed variable range for methods vfb_rgb, vfb_rgba, vfb (width and height) are now int instead 
   of unsigned int. Changed the test_mapping.py to reflect those changes  
+  corrected profiling to be run from command line 

"""


# MAP BUFFER INDEX VALUE INTO 3D INDEXING
cpdef tuple to3d(unsigned int index, unsigned int width, unsigned short int depth):
    """
    Index mapping (buffer indexing --> 3d array)
    
    Knowing the index value of a buffer (index), find the equivalent position in a
    3d array (x, y, z) such as :
    buffer[index] --> 3d array[x, y, z]
    
    e.g :
    # Build a 3d array using the function to3d
    for i in range(length):
        x, y, z = to3d(i, w, 3)
        rgb_array[x, y, z] = c_buffer[i]
    
    :param index: python int; buffer index value in range [0...4294967295] 
    :param width: python int; width (3d array columns number) value in range [0...4294967295] 
    :param depth: python int; depth (RGB = 3) | (RGBA = 4) value in range [0...65535]
    :return     : Return a python tuple containing x, y, z index values 
    """
    cdef xyz v = to3d_c(index, width, depth)
    return v.x, v.y, v.z


# MAP 3D INDEX VALUE INTO BUFFER INDEXING
cpdef unsigned int to1d(unsigned int x, unsigned int y,
                        unsigned int z, unsigned int width, unsigned short int depth):
    """
    Index mapping (3d array indexing --> buffer)
    
    Knowing the index values of a 3d array (x, y, z), find the equivalent index position in a
    1d array (C buffer data type) such as :   
    3d array[x, y, z] --> buffer[index]
    
    e.g 
    # Convert 3d array (rgb_array) into a C buffer (1d)
    for i in range(w):
        for j in range(h):
            for k in range(3):
                index = to1d(i, j, k, w, 3)
                c_buffer[index] = rgb_array[i, j, k]       
                
    * Both arrays must have the same length 
    
    :param x     : python int; index x of the array in range [0 ... 4294967295] such as array[x, y, z]
    :param y     : python int; index y of the array in range [0 ... 4294967295] such as array[x, y, z]
    :param z     : python int; index z of the array in range [0 ... 4294967295] such as array[x, y, z]
    :param width : python int; width of the 3d array (number of columns) in range [0 ...  4294967295]. 
    If the 3d array is build from a pygame.Surface, then width is also the image width
    :param depth : python int; depth, either RGB (depth = 3) or RGBA (depth = 4)
    :return      : python int; return the index value (1d array) corresponding to a 3d array with index position 
    (x, y, z) The index value is cap to [0 ... 4294967295]
    """
    return to1d_c(x, y, z, width, depth)

# VERTICALLY FLIP A SINGLE BUFFER VALUE
cpdef vmap_buffer(unsigned int index, unsigned int width, unsigned int height, unsigned short int depth):
    """
    Vertically flipped a single buffer value.

    Flip a C-buffer value vertically
    Re-sample a buffer value in order to swap rows and columns of its equivalent 3d model

    Here is a 9 pixels buffer (length = 27), pixel format RGB

    buffer = [RGB1, RGB2, *RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, RGB9]
    Equivalent 3d model would be (3x3x3):
    3d model = [RGB1 RGB2 *RGB3*]
               [RGB4 RGB5 RGB6]
               [RGB7 RGB8 RGB9]

    below flipped buffer
    buffer = [RGB1, RGB4, *RGB7*, RGB2, RGB5, RGB8, RGB3, RGB6, RGB9]

    Equivalent 3d model flipped 
    3D model = [RGB1, RGB4, RGB7]
               [RGB2, RGB5, RGB8]
               [RGB3, RGB6, RGB9]

    output index value should be *RGB7* = 2

    :param index  : integer; index value to convert . Must be in range [0, 4294967295]
    :param width  : integer; Original image width . Must be in range [0, 4294967295]
    :param height : integer; Original image height . Must be in range [0, 4294967295]
    :param depth  : integer; Original image depth=3 for RGB or 4 for RGBA . Must be in range [0, 65535]
    :return       : integer value pointing to the pixel in the buffer (traversed vertically). 
    """
    return vmap_buffer_c(index, width, height, depth)

# Todo this could be done inplace
# FLIP VERTICALLY A BUFFER (TYPE RGB)
cpdef np.ndarray[np.uint8_t, ndim=1] vfb_rgb(
        unsigned char [:] source, unsigned char [:] target, int width, int height):
    """
    Vertically flipped buffer containing any format of RGB colors
    
    Flip a C-buffer vertically filled with RGB values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGB otherwise a ValueError
    will be raised.
    SOURCE AND TARGET ARRAY MUST BE SAME SIZE.
    This method is using Multiprocessing OPENMP if enabled during the compilation
    
    e.g
    Here is a 9 pixels buffer (length = 27), pixel format RGB
    
    buffer = [RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, RGB9]
    Equivalent 3d model would be (3x3x3):
    3d model = [RGB1 RGB2 RGB3]
               [RGB4 RGB5 RGB6]
               [RGB7 RGB8 RGB9]

    After vbf_rgb:
    output buffer = [RGB1, RGB4, RGB7, RGB2, RGB5, RGB8, RGB3, RGB6, RGB9]
    and its equivalent 3d model
    3D model = [RGB1, RGB4, RGB7]
               [RGB2, RGB5, RGB8]
               [RGB3, RGB6, RGB9]
    
    :param source   : 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGB.      
    :param target   : Target buffer must have same length than source buffer)
    :param width    : integer; Source array's width (or width of the original image). 
    :param height   : integer; source array's height (or height of the original image). 
    :return         : Return a vertically flipped 1D RGB buffer (swapped rows and columns of the 2d model) 
    
    """
    assert width  > 0, 'Argument width cannot be <=0'
    assert height > 0, 'Argument height cannot be <=0'
    return numpy.asarray(vfb_rgb_c(source, target, width, height))

# TODO this could be done inplace
# FLIP VERTICALLY A BUFFER (TYPE RGBA)
cpdef np.ndarray[np.uint8_t, ndim=1] vfb_rgba(
        unsigned char [:] source, unsigned char [:] target, int width, int height):
    """
    Vertically flipped buffer containing any format of RGBA colors
    
    Flip a C-buffer vertically filled with RGBA values
    Re-sample a buffer in order to swap rows and columns of its equivalent 3d model
    For a 3d numpy.array this function would be equivalent to a transpose (1, 0, 2)
    Buffer length must be equivalent to width x height x RGBA otherwise a valuerror
    will be raised.
    SOURCE AND TARGET ARRAY MUST BE SAME SIZE.
    This method is using Multiprocessing OPENMP
    e.g
    Here is a 9 pixels buffer (length = 36), pixel format RGBA
    
    buffer = [RGBA1, RGBA2, RGBA3, RGBA4, RGBA5, RGBA6, RGBA7, RGBA8, RGBA9]
    Equivalent 3d model would be (3x3x4):
    3d model = [RGBA1 RGBA2 RGBA3]
               [RGBA4 RGBA5 RGBA6]
               [RGBA7 RGBA8 RGBA9]

    After vbf_rgba:
    output buffer = [RGB1A, RGB4A, RGB7A, RGB2A, RGB5A, RGBA8, RGBA3, RGBA6, RGBA9]
    and its equivalent 3d model
    3D model = [RGBA1, RGBA4, RGBA7]
               [RGBA2, RGBA5, RGBA8]
               [RGBA3, RGBA6, RGBA9]
        
    :param source   : 1d buffer to flip vertically (unsigned char values).
     The array length is known with (width * height * depth). The buffer represent 
     image 's pixels RGBA.     
    :param target   : Target buffer must have same length than source buffer)
    :param width    : integer; Source array's width (or width of the original image). 
    :param height   : integer; source array's height (or height of the original image). 
    :return         : Return a vertically flipped 1D RGBA buffer (swapped rows and columns of the 2d model) 
    """
    assert width > 0, 'Argument width cannot be <=0'
    assert height > 0, 'Argument height cannot be <=0'
    return numpy.asarray(vfb_rgba_c(source, target, width, height))

# TODO this could be done inplace
# FLIP VERTICALLY A BUFFER (TYPE ALPHA, (WIDTH, HEIGHT))
cpdef unsigned char [::1] vfb(unsigned char [:] source,
                              unsigned char [::1] target, int width, int height):
    """
    Flip vertically the content (e.g alpha values) of an 1d buffer structure.
    buffer representing an array type (w, h) 

    :param source: 1d buffer created from array type(w, h) 
    :param target: 1d buffer numpy.empty(ax_ * ay_, dtype=numpy.uint8) that will be the equivalent 
    of the source array but flipped vertically 
    :param width: source width. 
    :param height: source height. 
    :return: return 1d buffer (source array flipped)
    """
    assert width > 0, 'Argument width cannot be <=0'
    assert height > 0, 'Argument height cannot be <=0'
    return vfb_c(source, target, width, height)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef inline xyz to3d_c(unsigned int index, unsigned int width, unsigned short int depth)nogil:

    if width == 0:
        with gil:
            printf("\nArgument width cannot be null!")
            raise ValueError

    if depth == 0:
        with gil:
            printf("\nArgument depth cannot be null!")
            raise ValueError

    cdef:
        xyz v
        unsigned int ix = index // depth

    v.y = <int>(ix / width)
    v.x = <int>(ix % width)
    v.z = <int>(index % depth)
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned int to1d_c(unsigned int x, unsigned int y,
                       unsigned int z,  unsigned int width, unsigned short int depth)nogil:

    return <unsigned int>(y * width * depth + x * depth + z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(False)
cdef inline unsigned int vmap_buffer_c(unsigned int index,
                              unsigned int width, unsigned int height, unsigned short int depth)nogil:
    if width == 0:
        with gil:
            printf("\nArgument width cannot be null!")
            raise ValueError

    if depth == 0:
        with gil:
            printf("\nArgument depth cannot be null!")
            raise ValueError
    cdef:
        unsigned int ix
        unsigned int x, y, z

    ix = index // depth
    y = <unsigned int>(ix / width)
    x = ix % width
    z = index % depth
    return <unsigned int>(x * height * depth) + (depth * y) + z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [:] vfb_rgb_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:
    cdef:
        int i, j, k, index
        unsigned char [:] flipped_array = target

    for i in prange(0, height * 3, 3):
        for j in range(0, width):
            index = i + (height * 3 * j)
            for k in range(3):
                flipped_array[(j * 3) + (i * width) + k] =  <unsigned char>source[index + k]

    return flipped_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [:] vfb_rgba_c(unsigned char [:] source, unsigned char [:] target,
                                   int width, int height)nogil:


    cdef:
        int i, j, k, index, v
        unsigned char [:] flipped_array = target

    for i in prange(0, height * 4, 4):
        for j in range(0, width):
            index = i + (height * 4 * j)
            v = (j * 4) + (i * width)
            for k in range(4):
                flipped_array[v + k] =  <unsigned char>source[index + k]

    return flipped_array


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline unsigned char [::1] vfb_c(unsigned char [:] source,
                               unsigned char [::1] target, int width, int height)nogil:
    cdef:
        int i, j
        unsigned char [::1] flipped_array = target

    for i in prange(0, height):
        for j in range(0, width):
            flipped_array[j + (i * width)] =  <unsigned char>source[i + (height * j)]
    return flipped_array


