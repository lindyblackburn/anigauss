# Python ctypes wrapper and circle test for fast anisotropic 2-D Gaussian filter
# wrapper for C library by J. M. Geusebroek: http://staff.science.uva.nl/~mark
# compile original C function as: gcc -fPIC -shared -o anigauss.so anigauss.c
#
# Lindy L Blackburn (lindylam@gmail.com)
# Jun 10, 2014

import os
import numpy as np
import ctypes as C

# load in external library
aglib = C.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "anigauss.so"))

# inarr = input array
# x-axis = inner index (most rapidly varying index)
# y-axis = outer index
# v-axis = short axis
# u-aixs = long axis, from which phi is defined
# (sigv, sigu) = sigmas for Gaussian filter
# phi = orientation angle in degrees (from u to x)
# (derv, deru) = for line and edge detection (default zero)
def anigauss(inarr, sigv, sigu, phi=0., derv=0, deru=0):
    
    # make sure we have a C-order array of doubles
    if inarr.dtype is not np.double:
        inarr = inarr.astype(np.double, order='C')
    elif not inarr.flags.c_contiguous:
        inarr = np.ascontiguousarray(inarr)

    # create output array
    outarr = np.zeros_like(inarr, dtype=np.double)

    # size parameters for array
    (sizey, sizex) = inarr.shape

    # call external function
    aglib.anigauss(inarr.ctypes.data_as(C.POINTER(C.c_double)),
                   outarr.ctypes.data_as(C.POINTER(C.c_double)),
                   C.c_int(sizex), C.c_int(sizey), C.c_double(sigv), C.c_double(sigu),
                   C.c_double(phi), C.c_int(derv), C.c_int(deru))

    # return filtered image
    return outarr

# plot circle and line in noise, and compare various Gaussian filters in 2D
def circletest():
    import matplotlib.pyplot as plt
    cm = plt.get_cmap('binary')
    kw = {'cmap':cm, 'interpolation':'none', 'origin':'upper'}

    (yy, xx) = np.mgrid[-50:61,-50:51]
    r = np.sqrt(xx**2 + yy**2)
    zz = np.exp(-(r - 36.)**2/8.) # hollow circle
    zz = np.maximum(zz, np.exp(-(xx+yy - 80.)**2/8.)) # line at x+y=80
    zz = np.maximum(zz, 0.5 * ((xx**2 + yy**2) < 20**2)) # filled circle
    zz += np.random.randn(*zz.shape) # noise
    plt.subplot(2, 3, 1)
    plt.imshow(zz, **kw)
    plt.title('original')

    plt.subplot(2, 3, 2)
    zzf = anigauss(zz, 2, 2)
    plt.imshow(zzf, **kw)
    plt.title('isotropic gaussian')

    plt.subplot(2, 3, 3)
    zz1 = anigauss(zz, 4, 4, derv=0, deru=2)
    zz2 = anigauss(zz, 4, 4, derv=2, deru=0)
    zzf = np.sqrt((zz1 + zz2)**2)
    plt.imshow(zzf, **kw)
    plt.title('laplace magnitude')

    zzfs = []
    zzds = []
    nangles = 32
    angles = np.arange(0, 180, 180./nangles)
    for phi in angles:

        zzf = anigauss(zz, 2, 6, phi=phi)
        zzd = anigauss(zz, 4, 6, derv=2, deru=0, phi=phi)
        zzfs.append(zzf)
        zzds.append(-zzd)

    plt.subplot(2, 3, 4)
    ishow = 4
    plt.imshow(zzfs[ishow], **kw)
    plt.title(u'anisotropic gaussian %d\u00b0' % angles[ishow])

    plt.subplot(2, 3, 5)
    plt.imshow(np.max(np.array(zzfs), axis=0), **kw)
    plt.title('maximum of %d anisotropic' % nangles)

    plt.subplot(2, 3, 6)
    plt.imshow(np.max(np.array(zzds), axis=0), **kw)
    plt.title('maximum of %d laplace' % nangles)
