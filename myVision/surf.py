#!/usr/bin/env python3
"""surf algorithm
ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf

"""

import itertools
import numpy as np
import scipy as sp
import scipy.misc
import scipy.signal ##convolution function
from scipy.ndimage.filters import convolve
from myVision.img_kernel import laplacian_gauss, laplacian_gauss, box_2nd_order
from myVision.img_kernel import *
from myVision.utils import rgba_to_grey, conv2, frobenius_norm, is_local_max

from numpy.linalg import det

#import skimage.color
#import matplotlib.colors as colors

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

__all__ = ['hello_surf', 'w', 'Surf']

Dx = np.asarray([1,-2,1])
Dxy = np.asarray([[1,-1],[-1,1]])
Dyx = np.asarray([[-1,1],[1,-1]])

OCTAVES = np.asarray( [[9,15,21,27],
                       [15,27,39,51],
                       [27,51,75,99],
                       [51,99,147,195]] )
#precalculated with generate_w script
HESSIAN_WEIGHTS = dict([(9, 0.91268594001907222), (15, 0.94869061617131789), (21, 0.96363318446943469), (27, 0.97183526068389059), (33, 0.97701885183554882), (39, 0.98059140382344656), (45, 0.98320300342889322), (51, 0.98519542446841735), (57, 0.98676553811928536), (63, 0.98803474925078982), (69, 0.98908199557940513), (75, 0.98996082350144599), (81, 0.99070883771330698), (87, 0.99135322299183781), (93, 0.99191411951583108), (99, 0.99240676595967259), (105, 0.99284290508799999), (111, 0.9932317318866104), (117, 0.99358054899412296), (123, 0.99389522969636179), (129, 0.99418055132593675), (135, 0.99444043950110117), (141, 0.99467814983528013), (147, 0.99489640502974208), (153, 0.99509749962807448), (159, 0.9952833809953372), (165, 0.99545571258791976), (171, 0.99561592387354025), (177, 0.99576525007603289), (183, 0.99590476408511397), (189, 0.99603540227578924), (195, 0.99615798555173685), (201, 0.99627323661256362), (207, 0.99638179421250717), (213, 0.99648422500489631), (219, 0.99658103343617965), (225, 0.996672670054224), (231, 0.99675953851967403), (237, 0.99684200155057856)])

"""
http://www.contrib.andrew.cmu.edu/~bpires/paperMedia/box_filters.pdf
"""

  
def w_sig(gauss_sig, box_size):
    #gaus_k = laplacian_gauss(gauss_sig,9)
    

    d_xy = box_2nd_order('xy',box_size)
    d_xx = box_2nd_order('xx',box_size)
    l_xx = gas_2nd_ord( 'xx', box_size)
    l_xy = gas_2nd_ord( 'xy', box_size)

    return frobenius_norm( l_xy) * frobenius_norm( d_xx ) / (frobenius_norm(l_xx) * frobenius_norm(d_xy))

def w(box_size):
    return w_sig(box_sigma(box_size), box_size)
    
    
class Surf:
    def __init__(self,th=400):
       self.hessianThreshold=th
    
    def _get_p(self,img, x, y):
        # return 0 if x or y is less than one
        if x < 0 or y < 0:
            return 0
        xdim, ydim = img.shape
        # solve the out of bounds problems
        return img[min(x,xdim-1),min(y,ydim-1)]
        
   
        
    def box_xx(self,i_img,ksize):
        xdim, ydim = i_img.shape
        l = int(ksize/3)
        h = int(l/2)
        out = np.zeros(i_img.shape)
        for i,j in itertools.product(range(xdim), range(ydim)):
            a1 = self._get_p(i_img, i-h-l-1, j-l)
            b1 = self._get_p(i_img, i-h-1  , j-l)
            c1 = self._get_p(i_img, i-h-l-1, j+l-1)
            d1 = self._get_p(i_img, i-h-1  , j+l-1)
            
            a2 = b1
            c2 = d1
            #a2 = self._get_p(i_img, i-h-1, j-l)
            b2 = self._get_p(i_img, i+h  , j-l)
            #c2 = self._get_p(i_img, i-h-1, j+l-1)
            d2 = self._get_p(i_img, i+h  , j+l-1)
            
            a3 = b2
            c3 = d2
            #a3 = self._get_p(i_img, i+h, j-l)
            b3 = self._get_p(i_img, i+h+l, j-l)
            #c3 = self._get_p(i_img, i+h  , j+l-1)
            d3 = self._get_p(i_img, i+h+l, j+l-1)
            out[i,j] = (a1-b1-c1+d1)-2*(a2-b2-c2+d2)+(a3-b3-c3+d3)
        return out            
            
            
    def box_xy(self,i_img,ksize):
        xdim, ydim = i_img.shape
        l = int(ksize/3)
        out = np.zeros(i_img.shape)
        for i,j in itertools.product(range(xdim), range(ydim)):
            a1 = self._get_p(i_img,i-l-1,j-l-1)
            b1 = self._get_p(i_img,i-1  ,j-l-1)
            c1 = self._get_p(i_img,i-l-1,j-1)
            d1 = self._get_p(i_img,i-1  ,j-1)
    
            a2 = self._get_p(i_img,i  ,j-l-1)
            b2 = self._get_p(i_img,i+l,j-l-1)
            c2 = self._get_p(i_img,i  ,j-1)
            d2 = self._get_p(i_img,i+l,j-1)
            
            a3 = self._get_p(i_img,i-l-1,j)
            b3 = self._get_p(i_img,i-1  ,j)
            c3 = self._get_p(i_img,i-l-1,j+l)
            d3 = self._get_p(i_img,i-1  ,j+l)
            
            a4 = self._get_p(i_img,i  ,j)
            b4 = self._get_p(i_img,i+l,j)
            c4 = self._get_p(i_img,i  ,j+l)
            d4 = self._get_p(i_img,i+l,j+l)
            
            out[i,j] = (a1-b1-c1+d1)-(a2-b2-c2+d2)-(a3-b3-c3+d3)+(a4-b4-c4+d4)
        return out
    
    def box_yy(self,i_img,ksize):
        xdim, ydim = i_img.shape
        l = int(ksize/3)
        h = int(l/2)
        out = np.zeros(i_img.shape)
        for i,j in itertools.product(range(xdim), range(ydim)):
                
            a1 = self._get_p(i_img,i-l  ,j-h-1-l)
            b1 = self._get_p(i_img,i+l-1,j-h-1-l)
            c1 = self._get_p(i_img,i-l  ,j-h-1)
            d1 = self._get_p(i_img,i+l-1,j-h-1)
            
            a2 = c1
            b2 = d1
            #a2 = self._get_p(i_img,i-l  ,j-h-1)
            #b2 = self._get_p(i_img,i+l-1,j-h-1)
            c2 = self._get_p(i_img,i-l  ,j+h)
            d2 = self._get_p(i_img,i+l-1,j+h)
            
            a3 = c2
            b3 = d2
            #a3 = self._get_p(i_img,i-l  ,j+h)
            #b3 = self._get_p(i_img,i+l-1,j+h)
            c3 = self._get_p(i_img,i-l  ,j+h+l)
            d3 = self._get_p(i_img,i+l-1,j+h+l)
            out[i,j] = (a1-b1-c1+d1)-2*(a2-b2-c2+d2)+(a3-b3-c3+d3)
        return out
    
    def det_hessian(self, img, box_size):
        """
        I am using the convolution kernel in numpy because it is faster than my
        crappy code. I need to write it in c. d'oh
        
        ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf pg 4
        """
        dxy = convolve(img, box_2nd_order('xy',box_size), mode='constant')
        dxx = convolve(img, box_2nd_order('xx',box_size), mode='constant')
        dyy = convolve(img, box_2nd_order('yy',box_size), mode='constant')
        return dxx*dyy - (HESSIAN_WEIGHTS[box_size]*dxy)**2
    
    def _d0(self, doh, x, y, p,l):
        dx = 1/(2*p)*(doh[x+p,y]-doh[x-p,y])
        dy = 1/(2*p)*(doh[x,y+p]-doh[x,y-p])
        dl = 1/(4*p)
    
    def is_refined(self, doh, x,y,l):
        p = 2**l
        #dx = 1/(2*p)*(doh[x+p,y]
        
    def find_points(self, doh, o, i):
        out_list = []
        ksize = (2*i+1)*3
        xdim,ydim = doh.shape
        th_mask = np.zeros(doh.shape)
        ## remove all borders
        th_mask[1:-1,1:-1] = (doh>self.hessianThreshold)[1:-1,1:-1]
        x_th, y_th = np.nonzero(th_mask)
        for x,y in zip(x_th,y_th):
            if is_local_max(doh,x,y):
                out_list.append((x,y,ksize))
        return out_list
                
            
    
    def detect_points(self, img, mask=None):
        """
        http://www.ipol.im/pub/art/2015/69/ page 93
        """
        doh = dict()
        #for ksize in OCTAVES.flatten():
        #    doh[ksize] = self.det_hessian(img, ksize)
        
        for o in range(1,4):
            for i in range(2,3):
                l = L(o,i)
                print( l)
        
        
    
def hello_surf():
    img2_filename = './sample_routines/resize_img/FOX_Sports_logo2.png'
    img2 = rgba_to_grey(sp.misc.imread( img2_filename ))
    #plt.imsave('/tmp/myImage.jpeg',img2)
    #log_k = laplacian_gauss(1.2,9)
    print( w_sig(1.2,9))
    surf = Surf(400)
    surf.detect_points(img2)


