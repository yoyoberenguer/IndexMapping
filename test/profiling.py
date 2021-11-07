"""
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

"""

import timeit

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

import os
import IndexMapping
from IndexMapping.mapping import to1d, to3d, vfb_rgb, vfb_rgba, vfb, vmap_buffer

PROJECT_PATH = IndexMapping.__path__
os.chdir(PROJECT_PATH[0] + "\\test")

if __name__ == '__main__':
    N = int(1e6)
    t = timeit.timeit("to1d(x=5, y=6, z=3, width=800, depth=3)", "from __main__ import to1d", number=N)
    print("Testing to1d per call %s overall time %s for %s" % (t / N, t, N))

    t = timeit.timeit("to3d(2, 800, 3)", "from __main__ import to3d", number=N)
    print("Testing to3d per call %s overall time %s for %s" % (t / N, t, N))

    N = int(1e3)
    size = 800 * 800 * 3
    source_buffer = numpy.empty(size, numpy.uint8)
    target_buffer = numpy.empty(size, numpy.uint8)
    for i in range(size):
        source_buffer[i] = i
        target_buffer[i] = i

    t = timeit.timeit("vfb_rgb(source_buffer, target_buffer, 800, 800)",
                      "from __main__ import vfb_rgb, source_buffer, target_buffer", number=N)
    print("Testing vfb_rgb per call %s overall time %s for %s" % (t / N, t, N))

    N = int(1e6)
    t = timeit.timeit("vmap_buffer(10, 64, 64, 3)",
                      "from __main__ import vmap_buffer", number=N)
    print("Testing vmap_buffer per call %s overall time %s for %s" % (t / N, t, N))

    N = int(1e3)
    size = 800 * 800 * 4
    source_buffer = numpy.empty(size, numpy.uint8)
    target_buffer = numpy.empty(size, numpy.uint8)
    for i in range(size):
        source_buffer[i] = i
        target_buffer[i] = i

    t = timeit.timeit("vfb_rgba(source_buffer, target_buffer, 800, 800)",
                      "from __main__ import vfb_rgba, source_buffer, target_buffer", number=N)
    print("Testing vfb_rgba per call %s overall time %s for %s" % (t / N, t, N))