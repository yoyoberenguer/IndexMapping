# cython: binding=False, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, optimize.use_switch=True
# encoding: utf-8


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


DEF M_VOID  = 0
DEF M_BOOL  = 1
DEF M_BYTE  = 2
DEF M_UBYTE = 3
DEF M_SHORT = 4
DEF M_USHORT= 5
DEF M_INT   = 6
DEF M_UINT  = 7
DEF M_HALF  = 8
DEF M_FLOAT = 9
DEF M_DOUBLE= 10

cdef extern from 'mapc.c' nogil:
    struct m_image:
       void *data;
       int size;
       int width;
       int height;
       int comp;
       char type;

    void m_image_create(m_image *image, char type_, int width, int height, int comp)
    void m_image_destroy(m_image *image);
    void m_flip_buffer(m_image *src, m_image *dest);
    void test_array_inplace(m_image *src)nogil;
    void test_rgb_inplace(m_image *src, m_image *red, m_image *green, m_image *blue)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef testing_pure_c(unsigned char [:] buffer_, int width, int height):

    cdef:
        m_image foo1;
        m_image foo2;
        int b_length, r;

    b_length = len(buffer_)

    # Create two buffers
    m_image_create(&foo1, M_UINT, width, height, 3)
    m_image_create(&foo2, M_UINT, width, height, 3)

    foo1.data = &buffer_[0]
    foo2.data = &buffer_[0]

    m_flip_buffer(&foo1, &foo2)

    image = <unsigned char *>foo1.data
    return image


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef test_c_inplace(unsigned char [:] buffer_, int width, int height):

    cdef:
        m_image foo1;
        int b_length, r;

    b_length = len(buffer_)

    # Create two buffers
    m_image_create(&foo1, M_UINT, width, height, 3)

    foo1.data = &buffer_[0]


    test_array_inplace(&foo1)

    image = <unsigned char *>foo1.data
    return image



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef rgb_inplace(unsigned char [:] buffer_, int width, int height):

    cdef:
        m_image rgb_array;
        m_image red_chanel;
        m_image green_channel;
        m_image blue_channel;

    # Create two buffers
    m_image_create(&rgb_array, M_UINT, width, height, 3)
    m_image_create(&red_chanel, M_UINT, width, height, 3)
    m_image_create(&green_channel, M_UINT, width, height, 3)
    m_image_create(&blue_channel, M_UINT, width, height, 3)

    cdef unsigned char [:] red_   = numpy.zeros(width * height * 3, numpy.uint8)
    cdef unsigned char [:] green_ = numpy.zeros(width * height * 3, numpy.uint8)
    cdef unsigned char [:] blue_  = numpy.zeros(width * height * 3, numpy.uint8)

    rgb_array.data      = &buffer_[0]
    red_chanel.data     = &red_[0]
    green_channel.data  = &green_[0]
    blue_channel.data   = &blue_[0]

    test_rgb_inplace(&rgb_array, &red_chanel, &green_channel, &blue_channel)

    r_surf = <unsigned char *> red_chanel.data
    g_surf = <unsigned char *> green_channel.data
    b_surf = <unsigned char *> blue_channel.data

    # m_image_destroy(&rgb_array)
    # m_image_destroy(&red_chanel)
    # m_image_destroy(&green_channel)
    # m_image_destroy(&blue_channel)

    return r_surf, g_surf, b_surf