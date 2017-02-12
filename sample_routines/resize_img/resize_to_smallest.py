#!/usr/bin/env python3

import numpy as np
import scipy as sp
import scipy.misc

img1_filename = '250px-Fox_Sports_logo.png'
img2_filename = 'FOX_Sports_logo2.png'
def main():
    img1 = sp.misc.imread( img1_filename )
    img2 = sp.misc.imread( img2_filename )
    if img1.size > img2.size:
        outimg = sp.misc.imresize( img1, img2.shape)
    else:
        outimg = sp.misc.imresize( img2, img1.shape)
    sp.misc.imsave('smaller_img.png', outimg)
        
    
    
if __name__ == "__main__":
    main()
