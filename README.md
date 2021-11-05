# IndexMapping 

```

This library contains tools that helps indexing arrays from 
`1d array` (C buffer data type) to 3d array and reciprocally.
The functions to1d and to3d are complementary and can be
resume to :

If we know the index value of a buffer (index), find the 
equivalent position in a 3d array (x, y, z) such as :

* to3d (buffer[index]     --> 3d array[x, y, z])
* to1d (3d array[x, y, z] --> buffer[index])

to1d & to3d can also be used to convert 1d array -> 3d array
and vice versa. The library also includes functions to transpose
C buffer data type like the numpy transpose function used for 
multi-dimensional arrays.

How can we transpose row and columns in a 1d array since 
a buffer is a contiguous or non-contiguous adjacent set of data?

1) reshape the 1d array to a 3d equivalent format 
2) swap row and column of the equivalent 3d array (transposed) 
3) Convert the 3d array back to 1d array type (flatten)

This library provide functions such as vfb_rgb & vfb_rgba 
(for transparency) to transpose the array directly.

Below a good example with 9 pixels buffer:
// Original 9 pixels buffer (length = 27), pixel format RGB 
(contiguous values)
buffer = [RGB1, RGB2, *RGB3, RGB4, RGB5, RGB6, RGB7, RGB8, 
RGB9]

Equivalent reshape model (w, h, depth) would be (3x3x3):
3d model = [RGB1 RGB2 *RGB3*]
           [RGB4 RGB5  RGB6 ]
           [RGB7 RGB8  RGB9 ]

// Same 1d buffer after transposing the values (swapped 
row and column)
buffer = [RGB1, RGB4, *RGB7*, RGB2, RGB5, RGB8, RGB3, RGB6, 
RGB9]

Equivalent reshape model (w, h, depth) after transposing the
original buffer
3D model = [RGB1, RGB4, RGB7]
           [RGB2, RGB5, RGB8]
           [RGB3, RGB6, RGB9]

After transposing the buffer we can observed that the 3d 
equivalent model is an array with row & columns swapped. 
This operation would be identical to a numpy transpose 
function such as : 3darray.transpose(1, 0, 2)

```


## Installation 
```
pip install Mapping
```

## Methods
```cython
# MAP BUFFER INDEX VALUE INTO 3D INDEXING
cpdef tuple to3d(unsigned int index, unsigned int width,
                 unsigned short int depth):
    """
    Index mapping (buffer indexing --> 3d array)
    
    :param index : python int; buffer index    
    :param width : python int; width (3d array columns number) 
    :param depth : python int; depth (RGB = 3) | (RGBA = 4) 
    :return      : Return a python tuple containing x, y, z index values 
    """
    cdef xyz v;
    v = to3d_c(index, width, depth)
    return v.x, v.y, v.z
    
# MAP 3D INDEX VALUE INTO BUFFER INDEXING
cpdef unsigned int to1d(
    unsigned int x, unsigned int y,
    unsigned int z, unsigned int width, unsigned short int depth):
    """
    Index mapping (3d array indexing --> buffer)
    
    :param x     : python int; index x of the array 
    :param y     : python int; index y of the array 
    :param z     : python int; index z of the array 
    :param width : python int; width of the 3d array (number of columns)
    :param depth : python int; depth, either RGB (depth = 3) or RGBA (depth = 4)
    :return      : python int; return the index value (1d array)
     corresponding to a 3d array with index position  (x, y, z) 
    """
    return to1d_c(x, y, z, width, depth)

# VERTICALLY FLIP A SINGLE BUFFER VALUE
cpdef vmap_buffer(unsigned int index, unsigned int width, 
                  unsigned int height, unsigned short int depth):
    """
    Vertically flipped a single buffer value.

    Flip a C-buffer value vertically
    Re-sample a buffer value in order to swap rows and columns of
    its equivalent 3d model

    :param index  : integer; index value to convert  
    :param width  : integer; Original image width 
    :param height : integer; Original image height   
    :param depth  : integer; Original image depth=3 for RGB or 4 for RGBA
    :return       : integer value pointing to the pixel in
     the buffer (traversed vertically). 
    """
    return vmap_buffer_c(index, width, height, depth)


# FLIP VERTICALLY A BUFFER (TYPE RGB)
cpdef np.ndarray[np.uint8_t, ndim=1] vfb_rgb(
        unsigned char [:] source, unsigned char [:] target,
        unsigned int width, unsigned int height):
    """
    Vertically flipped buffer containing any format of RGB colors
    
    :param source   : 1d buffer to flip vertically (unsigned char 
    values). The array length is known with (width * height *
    depth). The buffer represent  image 's pixels RGB.      
    :param target   : Target buffer must have same length than source  buffer)
    :param width    : integer; Source array's width (image width). 
    :param height   : integer; source array's height (image width). 
    :return         : Return a vertically flipped 1D RGB buffer 
    (swapped rows and columns of the 2d model) 
    
    """
    return numpy.asarray(vfb_rgb_c(source, target, width, height))

# FLIP VERTICALLY A BUFFER (TYPE RGBA)
cpdef np.ndarray[np.uint8_t, ndim=1] vfb_rgba(
        unsigned char [:] source, unsigned char [:] target,
        unsigned int width, unsigned int height):
    """
    Vertically flipped buffer containing any format of RGBA colors
        
    :param source   : 1d buffer to flip vertically (unsigned 
    char values). The array length is known with (width * height
     * depth). The buffer represent image 's pixels RGBA.     
    :param target   : Target buffer must have same length than source buffer)
    :param width    : integer; Source array's width (image width)
    :param height   : integer; source array's height (image height)
    :return         : Return a vertically flipped 1D RGBA buffer 
    (swapped rows and columns of the 2d model) 
    """
    return numpy.asarray(vfb_rgba_c(source, target, width, height))


# FLIP VERTICALLY A BUFFER (TYPE ALPHA, (WIDTH, HEIGHT))
cpdef unsigned char [::1] vfb(
    unsigned char [:] source,
    unsigned char [::1] target, unsigned int width,
    unsigned int height):
    """
    Flip vertically the content (e.g alpha values) of an 1d buffer
    structure. buffer representing an array type (w, h) 

    :param source : 1d buffer created from array type(w, h) 
    :param target : 1d buffer numpy.empty(ax_ * ay_, dtype=numpy.uint8) 
    that will be the equivalentof the source array but flipped vertically 
    :param width  : source width. 
    :param height : source height. 
    :return       : return 1d buffer (source array flipped)
    """
    return vfb_c(source, target, width, height)
    
```

``` python

EXAMPLE :

from IndexMapping.mapping import to1d
import pygame

width, height = 800, 1024
screen = pygame.display.set_mode((width, height))
background = pygame.image.load('Assets/A1.png').convert()
w, h = background.get_size()
rgb_array = pygame.surfarray.pixels3d(background)
c_buffer = numpy.empty(w * h * 3, dtype=numpy.uint8)

# Convert 3d array (rgb_array) into a C buffer (1d)
for i in range(w):
    for j in range(h):
        for k in range(3):
            index = to1d(i, j, k, w, 3)
            c_buffer[index] = rgb_array[i, j, k]       


from IndexMapping.mapping import to3d
import pygame
from pygame.surfarray import pixels3d
import numpy

width, height = 800, 1024
screen = pygame.display.set_mode((width, height))
background = pygame.image.load('Assets/A1.png').convert()
w, h = background.get_size()

rgb_array = pixels3d(background).transpose(1, 0, 2)
c_buffer = rgb_array.flatten()

length = c_buffer.size
assert length == w * h * 3, \
    "C buffer has an incorrect length, got %s instead of %s " \ 
% (length, w * h * 3)

rgb_array = numpy.zeros((w, h, 3), numpy.uint8)
        
# Build a 3d array using the function to3d
for i in range(length):
    x, y, z = to3d(i, w, 3)
    rgb_array[x, y, z] = c_buffer[i]
```
```python
import numpy
from IndexMapping.mapping import vfb_rgb
import pygame
from pygame.surfarray import pixels3d

width, height = 800, 1024
screen = pygame.display.set_mode((width, height))
background = pygame.image.load('Assets/A1.png').convert()
background = pygame.transform.smoothscale(background, (640, 480))
w, h = background.get_size()
rgb_array = pixels3d(background)

rgb_buffer = rgb_array.flatten()
target_buffer = numpy.empty(w * h * 3, numpy.uint8)
rgb_buffer_transpose = vfb_rgb(rgb_buffer, target_buffer, w, h)
```

## Building cython code
```
If you need to compile the Cython code after any changes in the 
file Mapping.pyx:

1) open a terminal window
2) Go in the main project directory where (mapping.pyx & 
   mapping.pxd files are located)
3) run : python setup_Mapping.py build_ext --inplace

If you have to compile the code with a specific python 
version, make sure to reference the right python version 
in (python38 setup_mapping.py build_ext --inplace)

If the compilation fail, refers to the requirement section and 
make sure cython and a C-compiler are correctly install on your
 system.
- A compiler such visual studio, MSVC, CGYWIN setup correctly on 
  your system.
  - a C compiler for windows (Visual Studio, MinGW etc) install 
  on your system and linked to your windows environment.
  Note that some adjustment might be needed once a compiler is 
  install on your system, refer to external documentation or 
  tutorial in order to setup this process.e.g https://devblogs.
  microsoft.com/python/unable-to-find-vcvarsall-bat/
```

## Credit
Yoann Berenguer 

## Dependencies :
```
python >= 3.0
cython >= 0.28
```

## License :
```
MIT License

Copyright (c) 2019 Yoann Berenguer

Permission is hereby granted, free of charge, to any person 
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without 
restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following 
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
```

## Timing :
```
Testing to1d        per call 2.627551e-07 
overall time 0.2627551 for 1000000

Testing to3d        per call 1.217256e-07 
overall time 0.1217255 for 1000000

Testing vfb_rgb     per call 0.0015129453 
overall time 1.5129453 for 1000       --> image 800x800x3

Testing vmap_buffer per call 1.189032e-07 
overall time 0.1189032 for 1000000

Testing vfb_rgba    per call 0.001878    
 overall time 1.8783595 for 1000       --> image 800x800x4
```