"""Gauss kernel and Gauss filter

"""

import numpy as np

__all__ = [gauss_kernel]

"""
similar to matlab fspecial("gaussian",sig)
"""

def gauss_kernel(sig):
   x, y = np.mgrid[-3*sig:(3*sig+1),-3*sig:(3*sig+1)]
   exponents = np.exp( -(x**2+y**2)/(2*sig**2))
   return exponents/ np.sum(exponents.flatten())
