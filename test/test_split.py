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
# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

import timeit

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, make_surface
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")

import os
import IndexMapping
from IndexMapping.mapcfunctions import testing_pure_c, test_c_inplace, rgb_inplace

PROJECT_PATH = IndexMapping.__path__
os.chdir(PROJECT_PATH[0] + "\\test")

def run_test_split():

    w, h = 800, 1024
    screen = pygame.display.set_mode((w * 2, h))

    # TESTING RGB SPLIT
    background = pygame.image.load('../Assets/A1.png').convert()
    background = pygame.transform.smoothscale(background, (w, h))
    background_rgb = pygame.surfarray.array3d(background)

    CLOCK = pygame.time.Clock()
    timer = 0
    while 1:
        pygame.event.pump()
        background_b = background_rgb.flatten()
        red, green, blue = rgb_inplace(background_b.astype(dtype=numpy.uint8), 800, 1024)
        red_surface = make_surface(numpy.frombuffer(red, dtype=numpy.uint8).reshape(w, h, 3))
        green_surface = make_surface(numpy.frombuffer(green, dtype=numpy.uint8).reshape(w, h, 3))
        blue_surface = make_surface(numpy.frombuffer(blue, dtype=numpy.uint8).reshape(w, h, 3))
        screen.fill((0, 0, 0))
        screen.blit(red_surface, (0, 0))
        screen.blit(green_surface, (20, 20), special_flags=pygame.BLEND_RGB_ADD)
        screen.blit(blue_surface, (20, 20), special_flags=pygame.BLEND_RGB_ADD)
        if timer > int(1e2):
            break
        timer += 1
        pygame.display.flip()
        CLOCK.tick()


if __name__ == '__main__':
    pass