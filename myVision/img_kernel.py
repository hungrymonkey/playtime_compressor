"""Gauss kernel and Gauss filter

   Note: numpy import images height by width
"""

import numpy as np

__all__ = ['gauss_kernel', 'gas_2nd_ord', 'gauss_2nd_order', 'hello_gauss','box_2nd_order','L', 'box_sigma']

"""
similar to matlab fspecial("gaussian",ksiz)
"""

def gauss_kernel(sig, ksize=3):
   l = int(ksize/2)
   x, y = np.mgrid[-l:l+1,-l:l+1]
   exponents = np.exp( -(x**2+y**2)/(2*sig**2))
   return exponents/exponents.sum()

def L(i):
    """
    http://www.ipol.im/pub/art/2015/69/ page 190
    """
    return 2*i+1
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
   
def box_1st_order( box, l ):
    dx = np.asarray([[1,1,1]])
    if box == "x":
        pass
    elif box == "y":
        pass
    else:
        raise ValueError('Invalid Input box needs to be either x or y')
        
def box_sigma( ksize):
    return .4*ksize/3


def gauss_2nd_order( sig, box, ksize=9 ):
   """
   Gxx=G(x,y,sig) = (-1+x^2/sig^2)* e^(-(x^2+y^2)/2/sig^2)/2*pi*sig^4
   Gyy=G(x,y,sig) = (-1+x^2/sig^2)* e^(-(x^2+y^2)/2/sig^2)/2*pi*sig^4
   Gxy=G(x,y,sig) = xy/2*pi*sig^6 * e^(-(x^2+y^2)/2/sig^2)
   http://campar.in.tum.de/Chair/HaukeHeibelGaussianDerivatives

   Alternate simple way
   http://stackoverflow.com/questions/23980080/derivative-of-gaussian-filter-in-matlab
   G1=fspecial('gauss',[round(k*sigma), round(k*sigma)], sigma);
   [Gx,Gy] = gradient(G1);   
   [Gxx,Gxy] = gradient(Gx);
   [Gyx,Gyy] = gradient(Gy);
   """
   l = int(ksize/2)
   x, y = np.mgrid[-l:l+1,-l:l+1]
   x2 = x**2
   y2 = y**2
   xy2 = x2+y2
   exponents = np.exp( -xy2/(2*sig**2) )
   sig6 = sig ** 6
   if box == "xx":
       return (x2 - sig ** 2)*exponents/( 2 * np.pi * sig**6)# * exponents.sum()) 
   elif box == "yy":
       return (y2 - sig ** 2)*exponents/( 2 * np.pi * sig**6)# * exponents.sum()) 
   elif box == "xy":
       numerator = ( x * y * exponents )
       return numerator/( 2 * np.pi * sig**6)# * exponents.sum()) 
   else:
      raise ValueError('Invalid Input box needs to be either xx, xy, or yy')

def gas_2nd_ord( box, ksize=9 ):
   return gauss_2nd_order( box_sigma( ksize), box, ksize )



def box_2nd_order( box, size=9 ):
   """
   ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf 
   Parameters:
   box : 
      This variable only accepts these strings: xx, yy, xy
      It creates a kernel such as [1,-2,1] [1,-2,1].T [1,-1;-1,1]
   size: This function is only tested on known surf octave such as
        9,15,21,27
   """
   c = int(size/2)
   
   """
   The box filter must grow at multiples of 2 pixels.
   """
   l = int(size/3)
   w = int(2*l-1)
   h = l-1
   box_filter = np.zeros((size,size))
   if box == "xx":
      e = int((size - 2*l-1)/2+1)
      box_filter[2*l:   , e:size-e] = 1
      box_filter[  l:2*l, e:size-e] = -2 #middle box
      box_filter[   :l  , e:size-e] = 1
   elif box == "yy":
      e = int((size - 2*l-1)/2+1)
      box_filter[e:size-e, 2*l:] = 1
      box_filter[e:size-e,   l:2*l] = -2 #middle box
      box_filter[e:size-e,    :l] = 1
   elif box == "xy":
      box_filter[c-l:c    ,c-l:c] =  1  #a
      box_filter[c+1:c+l+1,c-l:c] =  -1  #b
      box_filter[c-l:c    ,c+1:c+l+1] = -1  #c
      box_filter[c+1:c+l+1,c+1:c+l+1] = 1  #d
      #a = [(c-1, c-1), (c-1-l, c-1-l), (c-1-l, c-1), (c-1, c-1-l)]
      #b = [(c+1, c-1), (c+1+l, c-1), (c+1, c-1-l), (c+1+l, c-1-l)]
      #c = [(c-1, c+1), (c-1, c+1+l), (c-1-l, c+1), (c-1-l, c+1+l)]
      #d = [(c+1, c+1), (c+1+l, c+1+l), (c+1, c+1+l), (c+1+l, c+1)]
   else:
      raise ValueError('Invalid Input box needs to be either xx, xy, or yy')
   return box_filter


#def box_2nd_order( box, size=9 ):
   #"""
   #http://www.ipol.im/pub/art/2015/69/ page 185
   #Parameters:
   #box : 
      #This variable only accepts these strings: xx, yy, xy
      #It creates a kernel such as [1,-2,1] [1,-2,1].T [1,-1;-1,1]
   #size: This function is only tested on known surf octave such as
        #9,15,21,27
   #"""
   #c = int(size/2)
   #"""
   #The box filter must grow at multiples of 2 pixels.
   #"""
   #l = int(size /3)
   #box_filter = np.zeros((size,size))
   #if box == "xx":
      #h = int(l/2)
      #box_filter[c+h:c+h+l+1, c-l:c+l+1] = 1
      #box_filter[c-h:c+h+1  , c-l:c+l+1] = -2 #middle box
      #box_filter[c-h-l:c-h  , c-l:c+l+1] = 1
   #elif box == "yy":
      #h = int(l/2)
      #box_filter[c-l:c+l+1, c+h:c+h+l+1] = 1
      #box_filter[c-l:c+l+1, c-h:c+h+1] = -2 #middle box
      #box_filter[c-l:c+l+1, c-h-l:c-h] = 1
   #elif box == "xy":
      #box_filter[c-l:c    ,c-l:c] =  -1  #a
      #box_filter[c+1:c+l+1,c-l:c] =   1  #b
      #box_filter[c-l:c    ,c+1:c+l+1] =  1  #c
      #box_filter[c+1:c+l+1,c+1:c+l+1] = -1  #d
      ##a = [(c-1, c-1), (c-1-l, c-1-l), (c-1-l, c-1), (c-1, c-1-l)]
      ##b = [(c+1, c-1), (c+1+l, c-1), (c+1, c-1-l), (c+1+l, c-1-l)]
      ##c = [(c-1, c+1), (c-1, c+1+l), (c-1-l, c+1), (c-1-l, c+1+l)]
      ##d = [(c+1, c+1), (c+1+l, c+1+l), (c+1, c+1+l), (c+1+l, c+1)]
   #else:
      #raise ValueError('Invalid Input box needs to be either xx, xy, or yy')
   #return box_filter
   

def hello_gauss():
   print( "hello world")
   print( gas_2nd_ord( 'xy', 5 ))
   print( gas_2nd_ord( 'xy', 5 ).sum())

