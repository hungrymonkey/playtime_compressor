#!/usr/bin/env python3
"""sift algorithm

"""


import numpy as np
import scipy as sp
import scipy.misc
import scipy.signal ##convolution function
from myVision.img_kernel import laplacian_gauss, laplacian_gauss
from myVision.utils import rgba_to_grey, conv2, frobenius_norm
from myVision.img_kernel import *

from numpy.linalg import det
                                                                                                                                                        
#import skimage.color                                                                                                                                   
#import matplotlib.colors as colors                                                                                                                     
                                                                                                                                                        
import matplotlib.image as mpimg                                                                                                                        
import matplotlib.pyplot as plt                                                                                                                         
                                                                                                                                                        
#__all__ = ['hello_surf']   


class Sift:
    
    def __init__(self):
        self.th = 400
        self.ksize = 5
        self.sigma = 1.2
        self.k = 2
    
    def octaves(self,img):
        octs = []
        gas = [ gauss_kernel(self.sigma,self.ksize+i*self.k) for i in range(1,5) ]
        for o in range(4):
            _img = scipy.misc.imresize(img, 100/(o+1))
            l = [_img]
            for i in range(4):
                l.append(convolve(img, gas, mode='constant'))
            octs.append(l) 
        return octs    
        
    #def detect(img):
    #    return octaves
        
        
def hello_sift(img):
    sift = Sift()
    blur_imgs = sift.octaves(img)
    print( blur_imgs[1][1])
                
    
