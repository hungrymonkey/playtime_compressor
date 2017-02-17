#!/usr/bin/env python3


import numpy as np

__all__ = ['rgba_to_grey', 'rgb_to_grey']


"""
http://www.tannerhelland.com/3643/grayscale-image-algorithm-vb6/
https://docs.gimp.org/2.6/en/gimp-tool-desaturate.html
"""
def rgb_to_grey(rgb_img):
     return np.dot(rgb_img.astype(np.float), np.asarray([.21, .72, .07]))
#    return np.dot(rgb_img.astype(np.float), np.ones(3)/3.)
 
def rgba_to_grey(rgba_img):
     return np.dot(rgba_img[:,:,:-1].astype(np.float), np.asarray([.21, .72, .07]))
