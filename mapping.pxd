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

# C-structure to store 3d array index values
cdef struct xyz:
    int x;
    int y;
    int z;

cdef xyz to3d_c(unsigned int index, unsigned int width, unsigned short int depth)nogil

cdef unsigned int to1d_c(unsigned int x, unsigned int y,
                         unsigned int z, unsigned int width, unsigned short int depth)nogil

cdef unsigned int vmap_buffer_c(unsigned int index,
                                unsigned int width, unsigned int height, unsigned short int depth)nogil

cdef unsigned char [:] vfb_rgb_c(
        unsigned char [:] source, unsigned char [:] target, int width, int height)nogil

cdef unsigned char [:] vfb_rgba_c(
        unsigned char [:] source, unsigned char [:] target, int width, int height)nogil

cdef unsigned char [::1] vfb_c(unsigned char [:] source, unsigned char [::1] target,
                               int width, int height)nogil

