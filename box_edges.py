#!/usr/bin/env python3

import numpy as np



def xy(ksize=9):
    l = int(ksize/3)
    h = int(l/2)
    xy = np.zeros((ksize,ksize))
    i=int(ksize/2)
    j=int(ksize/2) 
    xy[i-l-1,j-l-1]=1
    xy[i-1  ,j-l-1]=1
    xy[i-l-1,j-1]=1
    xy[i-1  ,j-1]=1
        
    xy[i  ,j-l-1]=1
    xy[i+l,j-l-1]=1
    xy[i  ,j-1]=1
    xy[i+l,j-1]=1
                
    xy[i-l-1,j]=1
    xy[i-1  ,j]=1
    xy[i-l-1,j+l]=1
    xy[i-1  ,j+l]=1
                
    xy[i  ,j]=1
    xy[i+l,j]=1
    xy[i  ,j+l]=1
    xy[i+l,j+l]=1
    return xy

def xx(ksize=9):

    l = int(ksize/3)
    h = int(l/2)
    xx = np.zeros((ksize+1,ksize+1))
    i=int(ksize/2+1)
    j=int(ksize/2+1)
    xx[ i-h-l-1, j-l]=1
    xx[ i-h-1  , j-l]=1
    xx[ i-h-l-1, j+l-1]=1
    xx[ i-h-1  , j+l-1]=1

    #a2 = b1
    #c2 = d1
    xx[ i-h-1, j-l]=1
    xx[ i+h  , j-l]=1
    xx[ i-h-1, j+l-1]=1
    xx[ i+h  , j+l-1]=1

    #a3 = b2
    #c3 = d2
    xx[ i+h, j-l]=1
    xx[ i+h+l, j-l]=1
    xx[ i+h  , j+l-1]=1
    xx[ i+h+l, j+l-1]=1
    return xx

def yy(ksize=9):
    l = int(ksize/3)
    h = int(l/2)
    yy = np.zeros((ksize+1,ksize+1))
    i=int(ksize/2+1)
    j=int(ksize/2+1)
    yy[i-l  ,j-h-1-l]=1
    yy[i+l-1,j-h-1-l]=1
    yy[i-l  ,j-h-1]=1
    yy[i+l-1,j-h-1]=1
                
              #  a2 = c1
               # b2 = d1
    yy[i-l  ,j-h-1]=1
    yy[i+l-1,j-h-1]=1
    yy[i-l  ,j+h]=1
    yy[i+l-1,j+h]=1
                
               # a3 = c2
               # b3 = d2
    yy[i-l  ,j+h]=1
    yy[i+l-1,j+h]=1
    yy[i-l  ,j+h+l]=1
    yy[i+l-1,j+h+l]=1
    return yy

def main():
   print( xx(9))
   print( xy(15))
   print( yy(9))

if __name__ == "__main__":
    main()
