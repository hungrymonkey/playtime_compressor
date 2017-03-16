#!/usr/bin/env python

import myVision
import scipy as sp
import scipy.misc

def main():
    img2_filename = './sample_routines/resize_img/FOX_Sports_logo2.png'
    img2 = myVision.rgba_to_grey(sp.misc.imread( img2_filename ))
    myVision.hello_gauss()
    myVision.hello_sift(img2)
    

if __name__ == "__main__":
    main()

