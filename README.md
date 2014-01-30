anigauss
========

Python ctypes wrapper for fast anisotropic Gaussian filter library by J. M. Geusebroek

This package requires the original C library by J. M. Geusebroek,

  > wget http://staff.science.uva.nl/~mark/downloads/anigauss.c

which must be first compiled as a shared library,

  > gcc -fPIC -shared -o anigauss.so anigauss.c

and placed in the same directory as the Python wrapper ani.py

The Geusebroek library provides fast approximate 2-d anisotropic Gaussian
filtering of image data with arbitrary orientation, which is not available in
scipy.ndimage. This can also be used for anisotropic Gaussian KDE for point
data by first binning into a regular histogram at higher resolution. The
functions scale as O(N) and are much faster than direct convolution, and
generally faster than FFT methods.

  > import ani
  > ani.circletest()
