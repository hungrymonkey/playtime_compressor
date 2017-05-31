#!/usr/bin/python3
import sys, argparse
import numpy as np
import cv2


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
    kaze = cv2.KAZE_create()
    (kp1, des1) = kaze.detectAndCompute(source_grey, None)
    (kp2, des2) = kaze.detectAndCompute(test_grey, None)
    print("# kps: {}, descriptors: {}".format(len(kp1), des1.shape))
    print("# kps: {}, descriptors: {}".format(len(kp2), des2.shape))
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in xrange(len(matches))]

if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(False)
    main(sys.argv[1:])

