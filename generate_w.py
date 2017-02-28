#!/usr/bin/env python3

import myVision

def calculate_w(i):
    ksize = myVision.L(i) * 3
    return myVision.w(ksize)

def main():
    w = []
    for i in range(1,40):
        w.append((myVision.L(i) * 3, calculate_w(i)))
    print(w)
    
    
if __name__ == "__main__":
    main()
    
