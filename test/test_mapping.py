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
import time

# NUMPY IS REQUIRED
try:
    import numpy
except ImportError:
    raise ImportError("\n<numpy> library is missing on your system."
          "\nTry: \n   C:\\pip install numpy on a window command prompt.")

import timeit
import unittest

# PYGAME IS REQUIRED
try:
    import pygame
    from pygame import Color, Surface, SRCALPHA, RLEACCEL, BufferProxy
    from pygame.surfarray import pixels3d, array_alpha, pixels_alpha, array3d, make_surface
    from pygame.image import frombuffer

except ImportError:
    raise ImportError("\n<Pygame> library is missing on your system."
          "\nTry: \n   C:\\pip install pygame on a window command prompt.")


from mapping import to1d, to3d, vfb_rgb, vfb_rgba, vfb, vmap_buffer


class Test_to1d(unittest.TestCase):

    def runTest(self) -> None:
        # Check the type (int)
        value = to1d(x=5, y=6, z=3, width=800, depth=3)
        x, y, z = 5, 6, 3
        depth = 3
        width = 800
        m = y * width * depth + x * depth + z
        self.assertIsInstance(value, int)
        self.assertEqual(value, m)
        self.assertRaises(OverflowError, to1d, -5, 6, 3, 800, 3)
        self.assertRaises(OverflowError, to1d, 5, -6, 3, 800, 3)
        self.assertRaises(OverflowError, to1d, 5, 6, -3, 800, 3)
        self.assertRaises(OverflowError, to1d, 5, 6, 3, -800, 3)
        self.assertRaises(OverflowError, to1d, 5, 6, 3, 800, -3)
        self.assertRaises(OverflowError, to1d, 5, 6, 3, 800 , 65535 + 1)
        self.assertRaises(OverflowError, to1d, 5, 6, 3, 4294967295 + 1, 65535)
        self.assertRaises(OverflowError, to1d, 5, 6, 4294967295 + 1, 4294967295, 65535)
        self.assertRaises(OverflowError, to1d, 5, 4294967295 + 1, 4294967295, 4294967295, 65535)
        self.assertRaises(OverflowError, to1d, 4294967295 + 1, 4294967295, 4294967295, 4294967295, 65535)
        t = to1d(4294967295, 4294967295, 4294967295, 4294967295, 65535)
        self.assertLess(t, 4294967295+1)
        x, y, z = 4294967295, 4294967295, 4294967295
        depth = 65535
        width = 4294967295
        m = y * width * depth + x * depth + z
        self.assertGreater(m, 4294967295)

        w, h, depth = 800, 600, 3

        for x in range(w):
            for y in range(h):
                for z in range(depth):
                    index_ = to1d(x, y, z, w, depth)
                    x_, y_, z_ = to3d(index_, w, depth)
                    self.assertEqual(x_, x)
                    self.assertEqual(y_, y)
                    self.assertEqual(z_, z)


class Test_display_to1d_array(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 1024
        screen = pygame.display.set_mode((width * 2, height))

        background = pygame.image.load('../Assets/A1.png').convert()
        w, h = background.get_size()
        rgb_array = pygame.surfarray.pixels3d(background)
        c_buffer = numpy.empty(w * h * 3, dtype=numpy.uint8)

        # Convert 3d array (rgb_array) into a C buffer (1d)
        for i in range(w):
            for j in range(h):
                index = to1d(i, j, 0, w, 3)
                c_buffer[index] = rgb_array[i, j, 0]
                c_buffer[index + 1] = rgb_array[i, j, 1]
                c_buffer[index + 2] = rgb_array[i, j, 2]

        # Convert the 3d array into a buffer and compare rgb_buffer to c_buffer (both 1d)
        rgb_buffer = rgb_array.flatten()
        assert all(rgb_buffer) == all(c_buffer)

        # TESTING TO1D WITH REAL IMAGE
        background = pygame.image.load('../Assets/A1.png').convert()
        w, h = background.get_size()

        background_rgb = pygame.surfarray.pixels3d(background)
        background_b = numpy.empty(w * h * 3, dtype=numpy.uint8)

        # Convert 3d array (rgb_array) into a C buffer (1d)
        for i in range(w):
            for j in range(h):
                for k in range(3):
                    index = to1d(i, j, k, w, 3)
                    background_b[index]     = background_rgb[i, j, k]
                    # background_b[index + 1] = background_rgb[i, j, 1]
                    # background_b[index + 2] = background_rgb[i, j, 2]

        new_surface = pygame.image.frombuffer(background_b, (w, h), "RGB")
        print(new_surface.get_size())

        # Display the image build from the C buffer
        print("Display the image after to1d processing")
        timer = time.time()
        while 1:
            pygame.event.pump()

            screen.fill((0, 0, 0))
            screen.blit(new_surface, (0, 0))

            if time.time() - timer > 5:
                break

            pygame.display.flip()


class Test_to3d(unittest.TestCase):

    def runTest(self) -> None:

        value = to3d(2, 800, 3)
        self.assertIsInstance(value, tuple)

        self.assertRaises(OverflowError, to3d, -2, 800, 3)
        self.assertRaises(OverflowError, to3d, 2, -800, 3)
        self.assertRaises(OverflowError, to3d, 2, 800, -3)
        self.assertRaises(OverflowError, to3d, 4294967295 + 1, 4294967295, 65535)
        self.assertRaises(OverflowError, to3d, 4294967295, 4294967295 + 1, 65535)
        self.assertRaises(OverflowError, to3d, 4294967295, 4294967295, 65535 + 1)

        x, y, z = to3d(2800, 800, 3)
        index = 2800
        depth = 3
        width = 800

        ix = index // depth
        yy = int(ix / width)
        xx = ix % width
        zz = index % depth

        self.assertEqual(xx, x)
        self.assertEqual(yy, y)
        self.assertEqual(zz, z)

        w, h, depth = 800, 600, 3
        length = w * h * 3
        index = w * 3

        x, y, z = to3d(index, w, depth)
        index_ = to1d(x, y, z, w, depth)
        self.assertEqual(index, index_)

        for r in range(length):
            x, y, z = to3d(r, w, depth)
            index_ = to1d(x, y, z, w, depth)
            self.assertEqual(r, index_)


class Test_display_to3d_array(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 800, 1024
        screen = pygame.display.set_mode((width, height))

        background = pygame.image.load('../Assets/A1.png').convert()
        w, h = background.get_size()

        rgb_array = pixels3d(background).transpose(1, 0, 2)
        rgb_array_flat = rgb_array.flatten()

        length = rgb_array_flat.size
        assert length == w * h * 3, "C buffer has an incorrect length, got %s instead of %s " % (length, w * h * 3)

        empty = numpy.zeros((w, h, 3), numpy.uint8)

        # Build the 3d array using the function to3d(i, w, 3)
        for i in range(length):
            x, y, z = to3d(i, w, 3)

            empty[x, y, z] = rgb_array_flat[i]

        new_surface = make_surface(empty)

        timer = time.time()
        print("Display the image after to3d processing")
        while 1:
            pygame.event.pump()

            screen.fill((0, 0, 0))
            screen.blit(new_surface, (0, 0))

            if time.time() - timer > 5:
                break

            pygame.display.flip()


class Test_vfb_rgb(unittest.TestCase):

    def runTest(self) -> None:
        # 3 * 3 * 3
        source_buffer = numpy.empty(27, numpy.uint8)
        target_buffer = numpy.empty(27, numpy.uint8)
        for i in range(27):
            source_buffer[i] = i
        for i in range(27):
            target_buffer[i] = i

        flipped_buffer = vfb_rgb(source_buffer, target_buffer, 3, 3)

        src_array3d = source_buffer.reshape(3, 3, 3).transpose(1, 0, 2)
        src_array_flat = src_array3d.flatten()

        self.assertTrue(numpy.array_equal(src_array_flat, flipped_buffer))

        # 32x32x3
        source_buffer = numpy.empty(32 * 32 * 3, numpy.uint8)
        target_buffer = numpy.empty(32 * 32 * 3, numpy.uint8)
        for i in range(32 * 32 * 3):
            source_buffer[i] = i
        for i in range(32 * 32 * 3):
            target_buffer[i] = i

        flipped_buffer = vfb_rgb(source_buffer, target_buffer, 32, 32)

        src_array3d = source_buffer.reshape(32, 32, 3).transpose(1, 0, 2)
        src_array_flat = src_array3d.flatten()

        self.assertTrue(numpy.array_equal(src_array_flat, flipped_buffer))
        self.assertIsInstance(flipped_buffer, numpy.ndarray)
        self.assertTrue(flipped_buffer.dtype, numpy.uint8)

        self.assertRaises(OverflowError, vfb_rgb, source_buffer, target_buffer, -32, 32)
        self.assertRaises(OverflowError, vfb_rgb, source_buffer, target_buffer, 32, -32)
        self.assertRaises(OverflowError, vfb_rgb, source_buffer, target_buffer, 4294967295 + 1, 4294967295)
        self.assertRaises(OverflowError, vfb_rgb, source_buffer, target_buffer, 4294967295, 4294967295 + 1)
        self.assertRaises(TypeError, vfb_rgb, [r for r in range(100)], target_buffer, 4294967295, 4294967295)
        self.assertRaises(TypeError, vfb_rgb, source_buffer, [r for r in range(100)], 4294967295, 4294967295)
        self.assertRaises(ValueError, vfb_rgb, numpy.empty((10, 10), numpy.uint8), target_buffer, 4294967295, 4294967295)
        self.assertRaises(ValueError, vfb_rgb, numpy.zeros(10, numpy.float32), target_buffer, 4294967295,
                          4294967295)


class Test_display_vfb_rgb(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 640, 480
        screen = pygame.display.set_mode((width * 2, height))

        background = pygame.image.load('../Assets/A1.png').convert()
        background = pygame.transform.smoothscale(background, (640, 480))
        w, h = background.get_size()
        rgb_array = pixels3d(background)
        rgb_array_t = numpy.ascontiguousarray(rgb_array.transpose(1, 0, 2))

        rgb_buffer = rgb_array.flatten()
        target_buffer = numpy.empty(w * h * 3, numpy.uint8)
        rgb_buffer_t = vfb_rgb(rgb_buffer, target_buffer, w, h)

        s1 = pygame.image.frombuffer(rgb_array_t, (w, h), 'RGB')
        s2 = pygame.image.frombuffer(rgb_buffer_t, (w, h), 'RGB')
        timer = time.time()
        while 1:

            pygame.event.pump()

            screen.fill((0, 0, 0))
            screen.blit(s1, (0, 0))
            screen.blit(s2, (640, 0))

            pygame.display.flip()

            if time.time() - timer > 5:
                break


class Test_vmap_buffer(unittest.TestCase):

    def runTest(self) -> None:

        source_buffer = numpy.empty(27, numpy.uint8)
        target_buffer = numpy.empty(27, numpy.uint8)
        for i in range(27):
            source_buffer[i] = i
        for i in range(27):
            target_buffer[i] = i

        flipped_buffer = vfb_rgb(source_buffer, target_buffer, 3, 3)

        src_array_flat = numpy.empty(27, numpy.uint8)
        for i in range(27):
            src_array_flat[i] = vmap_buffer(i, 3, 3, 3)

        self.assertTrue(numpy.array_equal(src_array_flat, flipped_buffer))

        source_buffer = numpy.empty(64 * 64 * 3, numpy.uint8)
        target_buffer = numpy.empty(64 * 64 * 3, numpy.uint8)
        for i in range(64 * 64 * 3):
            source_buffer[i] = i
        for i in range(64 * 64 * 3):
            target_buffer[i] = i

        flipped_buffer = vfb_rgb(source_buffer, target_buffer, 64, 64)

        src_array_flat = numpy.empty(64 * 64 * 3, numpy.uint8)
        for i in range(64 * 64 * 3):
            src_array_flat[i] = vmap_buffer(i, 64, 64, 3)

        self.assertTrue(numpy.array_equal(src_array_flat, flipped_buffer))

        self.assertRaises(OverflowError, vmap_buffer, 4294967295 + 1, 4294967295, 4294967295, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, 4294967295 + 1, 4294967295, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, 4294967295, 4294967295 + 1, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, 4294967295, 4294967295, 65535 + 1)
        self.assertRaises(OverflowError, vmap_buffer, -4294967295, 4294967295, 4294967295, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, -4294967295, 4294967295, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, 4294967295, -4294967295, 65535)
        self.assertRaises(OverflowError, vmap_buffer, 4294967295, 4294967295, 4294967295, -65535)
        value = vmap_buffer(10, 64, 64, 3)
        self.assertIsInstance(value, int)


class Test_display_vmap_buffer(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 640, 480
        screen = pygame.display.set_mode((width * 2, height))

        background = pygame.image.load('../Assets/A1.png').convert()
        background = pygame.transform.smoothscale(background, (640, 480))
        w, h = background.get_size()
        rgb_array = pixels3d(background)
        # rgb_array_t = numpy.ascontiguousarray(rgb_array.transpose(1, 0, 2))
        c_buffer = numpy.empty(w * h * 3, numpy.uint8)

        flat = rgb_array.flatten()
        for i in range(0, w * h * 3):
            index = vmap_buffer(i, w, h, 3)
            # x, y, z = to3d(index, w, 3)
            c_buffer[i] = flat[index]

        temp_buffer = numpy.empty(w * h * 3, numpy.uint8)
        new_buffer = vfb_rgb(rgb_array.flatten(), temp_buffer, w, h)

        s1 = pygame.image.frombuffer(c_buffer, (w, h), 'RGB')
        s2 = pygame.image.frombuffer(new_buffer, (w, h), 'RGB')

        timer = time.time()
        while 1:

            pygame.event.pump()

            screen.fill((0, 0, 0))
            screen.blit(s1, (0, 0))
            screen.blit(s2, (w, 0))

            pygame.display.flip()

            if time.time() - timer > 5:
                break


class Test_vfb_rgba(unittest.TestCase):

    def runTest(self) -> None:
        # vfb_rgba(unsigned char [:] source, unsigned char [:] target, unsigned int width, unsigned int height)
        width, height = 640, 480
        screen = pygame.display.set_mode((width * 2, height))
        background = pygame.image.load('../Assets/A1.png').convert_alpha()
        w, h = background.get_size()
        self.assertEqual(background.get_bitsize(), 32)
        self.assertEqual(background.get_bytesize(), 4)
        source_buffer = numpy.empty(27, numpy.uint8)
        target_buffer = numpy.empty(27, numpy.uint8)

        value = vfb_rgba(source_buffer, target_buffer, 3, 3)
        self.assertIsInstance(value, numpy.ndarray)

        self.assertRaises(OverflowError, vfb_rgba, source_buffer, target_buffer, -32, 32)
        self.assertRaises(OverflowError, vfb_rgba, source_buffer, target_buffer, 32, -32)
        self.assertRaises(OverflowError, vfb_rgba, source_buffer, target_buffer, 4294967295 + 1, 4294967295)
        self.assertRaises(OverflowError, vfb_rgba, source_buffer, target_buffer, 4294967295, 4294967295 + 1)
        self.assertRaises(TypeError, vfb_rgba, [r for r in range(100)], target_buffer, 4294967295, 4294967295)
        self.assertRaises(TypeError, vfb_rgba, source_buffer, [r for r in range(100)], 4294967295, 4294967295)
        self.assertRaises(ValueError, vfb_rgba, numpy.empty((10, 10), numpy.uint8), target_buffer, 4294967295,
                          4294967295)
        self.assertRaises(ValueError, vfb_rgba, numpy.zeros(10, numpy.float32), target_buffer, 4294967295, 4294967295)

        # Create a C buffer (1d) from the original image
        rgba_buffer = numpy.array(background.get_view('0'), numpy.uint8)
        length = rgba_buffer.size
        self.assertEqual(length, w * h * 4)

        # Create a buffer to build the image
        target_buffer = numpy.empty(length, numpy.uint8)

        # Transpose the buffer
        new_buffer = vfb_rgba(rgba_buffer, target_buffer, w, h)

        # Build equivalent buffer from original image (and transpose it)
        rgba_array = rgba_buffer.reshape(w, h, 4)
        rgba_array_t = rgba_array.transpose(1, 0, 2)
        rgba_array_flat = rgba_array_t.flatten()

        # Both buffer should be equivalent
        self.assertTrue(numpy.array_equal(rgba_array_flat, new_buffer))


class Test_display_vfb_rgba(unittest.TestCase):

    def runTest(self) -> None:
        width, height = 640, 480
        screen = pygame.display.set_mode((width * 2, height))

        background = pygame.image.load('../Assets/A1.png').convert_alpha()
        background = pygame.transform.smoothscale(background, (640, 480))
        w, h = background.get_size()

        # Create a C buffer (1d) from the original image
        rgba_buffer = numpy.array(background.get_view('0'), numpy.uint8)
        length = rgba_buffer.size
        self.assertEqual(length, w * h * 4)

        # Create a buffer to build the image
        target_buffer = numpy.empty(length, numpy.uint8)

        # Transpose the buffer
        new_buffer = vfb_rgba(rgba_buffer, target_buffer, w, h)

        # Build equivalent buffer from original image (and transpose it)
        rgba_array = rgba_buffer.reshape(w, h, 4)
        rgba_array_t = rgba_array.transpose(1, 0, 2)
        rgba_array_flat = rgba_array_t.flatten()

        # Both buffer should be equivalent
        self.assertTrue(numpy.array_equal(rgba_array_flat, new_buffer))

        print(rgba_array_flat)

        s1 = pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')
        s2 = pygame.image.frombuffer(rgba_array_flat, (w, h), 'RGBA')

        timer = time.time()
        while 1:

            pygame.event.pump()

            screen.fill((0, 0, 0))
            screen.blit(s1, (0, 0))
            screen.blit(s2, (w, 0))

            pygame.display.flip()

            if time.time() - timer > 5:
                break



if __name__ == '__main__':

    suite = unittest.TestSuite()

    suite.addTests([Test_to1d(),
                    Test_display_to1d_array(),
                    Test_to3d(),
                    Test_display_to3d_array(),
                    Test_vfb_rgb(),
                    Test_display_vfb_rgb(),
                    Test_vmap_buffer(),
                    Test_display_vmap_buffer(),
                    Test_vfb_rgba(),
                    Test_display_vfb_rgba()

    ])

    unittest.TextTestRunner().run(suite)




    # # background_b = background_rgb # .transpose(1, 0, 2)
    # # background_b = background_b.flatten()
    # # im = testing_pure_c(background_b.astype(dtype=numpy.uint8), 400, 400)

    # im = pygame.image.frombuffer(background_b, (w, h), 'RGB')
    # screen.fill((0, 0, 0))
    # screen.blit(im, (0, 0))
    #
    # new_buffer = numpy.asarray(background_b).reshape(h, w, 3).transpose(1, 0, 2)
    # im = pygame.surfarray.make_surface(new_buffer)
    # screen.blit(im, (w, 0))
    # pygame.display.flip()
    # print('\nTESTING TO1D')
    # pygame.time.wait(1000)
    #
    # for r in range(w * h * 3):
    #     v = to3d(r, w, 3)
    #     x, y, z = v['x'], v['y'], v['z']
    #     background_rgb[x][y][z] = background_b[r]
    #
    # im = pygame.surfarray.make_surface(background_rgb)
    # screen.fill((0, 0, 0))
    # screen.blit(im, (0, 0))
    # pygame.display.flip()
    # print('\nTESTING TO3D')
    # pygame.time.wait(1000)
    #
    # # TESTING VFB_RGB (RGB ARRAY FLIP)
    # # RELOADING RGB ARRAY
    # background_rgb = pygame.surfarray.pixels3d(background)
    # # FLAT THE ARRAY IN A BUFFER
    # background_flat = background_rgb.flatten()
    #
    # background_c = numpy.empty(w * h * 3, dtype=numpy.uint8)
    # # TRANSPOSE THE ARRAY
    # flipped = vfb_rgb(background_flat, background_c, w, h)
    # back = pygame.image.frombuffer(flipped, (w, h), 'RGB')
    # back1 = pygame.surfarray.make_surface(numpy.asarray(flipped).reshape(h, w, 3))
    # screen.fill((0, 0, 0))
    # screen.blit(back, (0, 0))
    # screen.blit(back1, (w, 0))
    # pygame.display.flip()
    # print('\nTESTING VFB_RGB')
    # pygame.time.wait(2000)
    #
    # # TESTING vfb_rgba
    # background = pygame.image.load('A1.png').convert_alpha()
    # background = pygame.transform.smoothscale(background, (w, h))
    # rgb_array = pygame.surfarray.pixels3d(background)
    # alpha = pygame.surfarray.pixels_alpha(background)
    # print(rgb_array.shape)
    # print(alpha.shape)
    # rgba_array = numpy.dstack((rgb_array, alpha)).astype(dtype=numpy.uint8)
    # print(rgba_array.shape)
    # print(rgba_array.flags)
    # surface = pygame.image.frombuffer(
    #     numpy.ascontiguousarray(rgba_array.transpose(1, 0, 2)), (w, h), 'RGBA')
    # target = numpy.empty((w * h * 4), dtype=numpy.uint8)
    # source = rgba_array.flatten()
    # surface = pygame.image.frombuffer(vfb_rgba(source, target, w, h), (w, h), 'RGBA')
    # i = 0
    #
    # # RETURN RGBA VALUES AND ARRAY SHAPE (WIDTH, HEIGHT)
    # buffer_ = background.get_view('2')
    # print(numpy.asarray(buffer_).shape, buffer_.length)
    #
    # # RETURN RGB VALUES AND ARRAY SHAPE (WIDTH, HEIGHT, 3)
    # buffer_ = background.get_view('3')
    # print(numpy.asarray(buffer_).shape, buffer_.length)
    #
    # buffer_ = background.get_view('2')
    # new_buffer = numpy.frombuffer(buffer_, dtype=numpy.uint8)
    # print(new_buffer.shape)
    # flipped_image = pygame.image.frombuffer(new_buffer, (w, h), 'RGBA')
    # # CREATE AND FLIP THE BUFFER
    # flipped_buffer = flipped_image.get_view('2')
    # flipped_buffer = numpy.frombuffer(flipped_buffer, numpy.uint8).reshape(w, h, 4)
    # flipped_buffer = flipped_buffer.transpose(1, 0, 2).flatten()
    # # LOAD THE ORIGINAL BUFFER AND FLIP IT TOO WITH vfb_rgba TO COMPARE BOTH BUFFER
    # target = numpy.empty((w * h * 4), numpy.uint8)
    # other_flipped_buffer = vfb_rgba(numpy.frombuffer(background.get_view('2'), numpy.uint8), target, w, h)
    # print(len(flipped_buffer), other_flipped_buffer.shape)
    # # COMPARE BOTH FLIPPED BUFFERS
    # for r in range(len(flipped_buffer)):
    #     if flipped_buffer[r] != other_flipped_buffer[r]:
    #         print(flipped_buffer[r], other_flipped_buffer[r])
    #         raise ValueError('\nArray are not identical')
    #
    # # TEST vfb_c
    # alpha_image = pygame.image.load('Radial4.png').convert_alpha()
    # alpha_image = pygame.transform.smoothscale(alpha_image, (w, h))
    # alpha_buffer = alpha_image.get_view('a')
    # print(alpha_buffer.length)
    # target = numpy.empty((w * h), numpy.uint8)
    # alpha_buffer = numpy.asarray(alpha_buffer)  # .transpose(1, 0)
    # alpha_buffer = alpha_buffer.flatten()
    # alpha_buffer_transposed = vfb(alpha_buffer, target, w, h)
    # alpha_buffer_original_transposed = numpy.asarray(
    #     alpha_image.get_view('a')).transpose(1, 0)
    # alpha_buffer_original_transposed = alpha_buffer_original_transposed.flatten()
    #
    # for r in range(len(alpha_buffer_transposed)):
    #     if alpha_buffer_transposed[r] != alpha_buffer_original_transposed[r]:
    #         print(alpha_buffer_transposed[r], alpha_buffer_original_transposed[r])
    #         raise ValueError('\nALPHA Array are not identical')
    #
    # N = w * h
    # print("TO1D ", timeit.timeit("to1d(1, 1, 0, 400, 3)", "from __main__ import to1d", number=N))
    # print("TO3D ", timeit.timeit("to3d(1, 400, 3)", "from __main__ import to3d", number=N))
    # N = 1000
    # print("VFB_RGB ", timeit.timeit("vfb_rgb(background_flat, background_c, 400, 400)",
    #                                 "from __main__ import vfb_rgb, background_flat, background_c", number=N) / N)
    # print("PURE C ", timeit.timeit("testing_pure_c(background_flat, 400, 400)",
    #                                "from __main__ import testing_pure_c, background_flat", number=N) / N)
    # target = numpy.empty((w * h * 4), numpy.uint8)
    # print("VFB_RGBA C ", timeit.timeit("vfb_rgba(source, target, w, h)",
    #                                    "from __main__ import vfb_rgba, source, target, w, h", number=N) / N)
    # target = numpy.empty((w * h), numpy.uint8)
    # print("VFB C ", timeit.timeit("vfb(alpha_buffer, target, w, h)",
    #                               "from __main__ import vfb, alpha_buffer, target, w, h", number=N) / N)
    #
    # while 1:
    #     pygame.event.pump()
    #     keys = pygame.key.get_pressed()
    #     for event in pygame.event.get():
    #         if event.type == pygame.MOUSEMOTION:
    #             MOUSE_POS = event.pos
    #
    #     if keys[pygame.K_F8]:
    #         pygame.image.save(screen, 'Screendump' + str(i) + '.png')
    #
    #     if keys[pygame.K_ESCAPE]:
    #         break
    #
    #     screen.fill((0, 0, 0))
    #
    #     screen.blit(flipped_image, (0, 0))
    #
    #     pygame.display.flip()
    #
    #     i += 1
    # pygame.quit()
