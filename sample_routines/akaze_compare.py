#!/usr/bin/python3
import sys, argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', nargs=1)
    ap.add_argument('-i', nargs=1)
    opts = ap.parse_args(argv)
    if not any([opts.s, opts.i]):
       ap.print_usage()
       quit()
    source_grey = cv2.imread(opts.s[0], cv2.IMREAD_GRAYSCALE)
    test_grey = cv2.imread(opts.i[0], cv2.IMREAD_GRAYSCALE)
    akaze = cv2.AKAZE_create()
    (kp1, des1) = akaze.detectAndCompute(source_grey, None)
    (kp2, des2) = akaze.detectAndCompute(test_grey, None)
    print("# kps: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("# kps: {}, descriptors: {}".format(len(kp2), des2.shape))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(source_grey,kp1,test_grey,kp2,matches[:10],None, flags=2)
    plt.imshow(img3),plt.show()

if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(False)
    main(sys.argv[1:])

