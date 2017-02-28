#!/usr/bin/env python3


import numpy as np
from scipy.ndimage.filters import convolve

__all__ = ['rgba_to_grey', 'rgb_to_grey','conv2', 'integral_img',
           'frobenius_norm']


"""
http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
https://docs.gimp.org/2.6/en/gimp-tool-desaturate.html
"""
def rgb_to_grey(rgb_img):
     return np.dot(rgb_img.astype(np.float), np.asarray([.21, .72, .07]))
#    return np.dot(rgb_img.astype(np.float), np.ones(3)/3.)
 
def rgba_to_grey(rgba_img):
     return np.dot(rgba_img[:,:,:-1].astype(np.float), np.asarray([.21, .72, .07]))

def frobenius_norm(mat):
    """
    https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    """
    return np.sqrt(np.power(mat,2).sum())

def integral_img(img):
    """
    cumsum([1,1,2,1,1]) = [1,2,4,5,6]
    """
    return img.cumsum(axis=0).cumsum(axis=1)
    

def conv2(x,y,mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    TODO: 
     - Support other modes than 'same' (see conv2.m)
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x,y, mode='constant', origin=origin)

    return z
