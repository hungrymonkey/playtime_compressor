#!/usr/bin/env python3
"""surf algorithm
ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

"""


import numpy as np
import scipy as sp
import scipy.misc
import scipy.signal ##convolution function
from myVision.img_kernel import laplacian_gauss, laplacian_gauss, box_2nd_order
from myVision.img_kernel import *
from myVision.utils import rgba_to_grey, conv2, frobenius_norm

from numpy.linalg import det

#import skimage.color
#import matplotlib.colors as colors

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

__all__ = ['hello_surf', 'w']

Dx = np.asarray([1,-2,1])
Dxy = np.asarray([[1,-1],[-1,1]])
Dyx = np.asarray([[-1,1],[1,-1]])

OCTAVES = np.asarray( [[9,15,21,27],
                       [15,27,39,51],
                       [27,51,75,99]] )
#precalculated with generate_w script
HESSIAN_WEIGHTS = dict([(9, 0.91268594001907222), (15, 0.94869061617131789), (21, 0.96363318446943469), (27, 0.97183526068389059), (33, 0.97701885183554882), (39, 0.98059140382344656), (45, 0.98320300342889322), (51, 0.98519542446841735), (57, 0.98676553811928536), (63, 0.98803474925078982), (69, 0.98908199557940513), (75, 0.98996082350144599), (81, 0.99070883771330698), (87, 0.99135322299183781), (93, 0.99191411951583108), (99, 0.99240676595967259), (105, 0.99284290508799999), (111, 0.9932317318866104), (117, 0.99358054899412296), (123, 0.99389522969636179), (129, 0.99418055132593675), (135, 0.99444043950110117), (141, 0.99467814983528013), (147, 0.99489640502974208), (153, 0.99509749962807448), (159, 0.9952833809953372), (165, 0.99545571258791976), (171, 0.99561592387354025), (177, 0.99576525007603289), (183, 0.99590476408511397), (189, 0.99603540227578924), (195, 0.99615798555173685), (201, 0.99627323661256362), (207, 0.99638179421250717), (213, 0.99648422500489631), (219, 0.99658103343617965), (225, 0.996672670054224), (231, 0.99675953851967403), (237, 0.99684200155057856)])

"""
http://www.contrib.andrew.cmu.edu/~bpires/paperMedia/box_filters.pdf
"""



def hessian(img):
    log_k = laplacian_gauss(1.2,9)
    lconv_img = sp.signal.convolve2d( img, log_k, mode='same')
    l_yy = sp.signal.convolve2d( lconv_img, Dx.T, mode='same')
    l_xy = sp.signal.convolve2d( lconv_img, Dxy, mode='same')
    l_xx = sp.signal.convolve2d( lconv_img, Dx, mode='same')
    l_yx = sp.signal.convolve2d( lconv_img, Dyx, mode='same')
    
    gaus_k = laplacian_gauss(9,9)
    gconv_img = sp.signal.convolve2d( img, gaus_k, mode='same')
    d_yy = sp.signal.convolve2d( gconv_img, Dx.T, mode='same')
    d_xy = sp.signal.convolve2d( gconv_img, Dxy, mode='same')
    d_xx = sp.signal.convolve2d( gconv_img, Dx, mode='same')
    d_yx = sp.signal.convolve2d( gconv_img, Dyx, mode='same')
    print( det( l_xy) * det( d_yy )/ (det(l_yy) * det(d_xy)))
    
def hessian_det(d_xx, d_yy, d_xy, box_size):
    w = HESSIAN_WEIGHTS[box_size]
    return (d_xx * dyy - ( w * d_xy )**2)/(box_size/3)**4
    
    
def w_sig(gauss_sig, box_size):
    #gaus_k = laplacian_gauss(gauss_sig,9)
    

    d_xy = box_2nd_order('xy',box_size)
    d_xx = box_2nd_order('xx',box_size)
    l_xx = gas_2nd_ord( 'xx', box_size)
    l_xy = gas_2nd_ord( 'xy', box_size)

    return frobenius_norm( l_xy) * frobenius_norm( d_xx ) / (frobenius_norm(l_xx) * frobenius_norm(d_xy))

def w(box_size):
    return w_sig(box_sigma(box_size), box_size)
    
def hello_surf():
    #img2_filename = './sample_routines/resize_img/FOX_Sports_logo2.png'
    #img2 = rgba_to_grey(sp.misc.imread( img2_filename ))
    #plt.imsave('/tmp/myImage.jpeg',img2)
    #log_k = laplacian_gauss(1.2,9)
    print( w_sig(1.2,9))


