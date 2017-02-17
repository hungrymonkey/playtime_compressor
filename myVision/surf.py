#!/usr/bin/env python3
"""surf algorithm

"""


import numpy as np
import scipy as sp
import scipy.misc
import scipy.signal ##convolution function
from myVision.gauss_kernel import laplacian_gauss, laplacian_gauss
from myVision.utils import rgba_to_grey

from numpy.linalg import det

#import skimage.color
#import matplotlib.colors as colors

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

__all__ = ['hello_surf']

Dx = np.asarray([1,-2,1])
Dxy = np.asarray([[1,-1],[-1,1]])
Dyx = np.asarray([[-1,1],[1,-1]])



def hessian(img):
    log_k = laplacian_gauss(1.2,9)
    lconv_img = sp.signal.convolve2d( img, log_k, mode='same')
    l_yy = sp.signal.convolve2d( lconv_img, Dx.T, mode='same')
    l_xy = sp.signal.convolve2d( lconv_img, Dxy, mode='same')
    l_xx = sp.signal.convolve2d( lconv_img, Dx, mode='same')
    l_yx = sp.signal.convolve2d( lconv_img, Dyx, mode='same')
    
    gaus_k = laplacian_gauss(1.2,9)
    gconv_img = sp.signal.convolve2d( img, gaus_k, mode='same')
    d_yy = sp.signal.convolve2d( gconv_img, Dx.T, mode='same')
    d_xy = sp.signal.convolve2d( gconv_img, Dxy, mode='same')
    d_xx = sp.signal.convolve2d( gconv_img, Dx, mode='same')
    d_yx = sp.signal.convolve2d( gconv_img, Dyx, mode='same')
    print( det( l_xy) * det( d_yy )/ (det(l_yy) * det(d_xy)))
    
def hello_surf():
    img2_filename = './sample_routines/resize_img/FOX_Sports_logo2.png'
    img2 = rgba_to_grey(sp.misc.imread( img2_filename ))
    plt.imsave('/tmp/myImage.jpeg',img2)
    #log_k = laplacian_gauss(1.2,9)
    print(hessian(img2))
