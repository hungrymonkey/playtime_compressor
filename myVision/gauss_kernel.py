"""Gauss kernel and Gauss filter

"""

import numpy as np

__all__ = ['gauss_kernel', 'hello_gauss']

"""
similar to matlab fspecial("gaussian",,ksiz)
"""

def gauss_kernel(sig, ksize=3):
   l = int(ksize/2)
   x, y = np.mgrid[-l:l+1,-l:l+1]
   exponents = np.exp( -(x**2+y**2)/(2*sig**2))
   return exponents/exponents.sum()
"""
similar to matlab 
pkg load image
fspecial("log",sig,ksiz)
https://www.mathworks.com/help/images/ref/fspecial.html#bu0j90n-2
"""
def laplacian_gauss(sig, ksize):
   l = int(ksize/2)
   x, y = np.mgrid[-l:l+1,-l:l+1]
   xy2 = x**2+y**2
   exponents = np.exp( -xy2/(2*sig**2) )
   return exponents * (xy2 - 2 * sig**2) / ( 2 * np.pi * sig**6 * exponents.sum()) 
   

def hello_gauss():
   print( "hello world")
