import cv2
import numpy as np

def condDilate(img, mask, kernel, anchor=(-1,-1)):
    last = img
    curr = img
    count = 0
    while True:
        count += 1
        curr = cv2.dilate(last, kernel, anchor=anchor)
        curr = np.min((curr,mask),axis=0)
        # curr = cv2.bitwise_and(curr, mask)
        diff = cv2.subtract(curr, last)
        if not np.any(diff):
            break
        if count>10000:
            print('out')
            break
        last = curr
    return curr

def geodesicDilate(img, mask, kernel, anchor=(-1,-1)):
    res = cv2.dilate(img, kernel, anchor=anchor)
    res = np.min((res, mask),axis=0)
    res = np.max((res, img),axis=0)
    return res

def geodesicErose(img, mask, kernel, anchor=(-1,-1)):
    res = cv2.erode(img, kernel, anchor=anchor)
    res = np.max((res, mask),axis=0)
    res = np.min((res, img),axis=0)
    return res

def reDilate(img, mask, kernel, anchor=(-1,-1)):
    last = img
    curr = img
    count = 0
    while True:
        count += 1
        curr = geodesicDilate(last, mask, kernel, anchor=anchor)
        diff = cv2.subtract(curr, last)
        if not np.any(diff):
            break
        if count>10000:
            print('out')
            break
        last = curr
    return curr

def reErose(img, mask, kernel, anchor=(-1,-1)):
    last = img
    curr = img
    count = 0
    while True:
        count += 1
        curr = geodesicErose(last, mask, kernel, anchor=anchor)
        diff = cv2.subtract(curr, last)
        if not np.any(diff):
            break
        if count>10000:
            print('out')
            break
        last = curr
    return curr

def openRe(img, kernel, anchor=(-1,-1)):
    res = cv2.erode(img, kernel, anchor=anchor)
    res = reDilate(res, img, kernel, anchor=anchor)
    return res

def closeRe(img, kernel, anchor=(-1,-1)):
    res = cv2.dilate(img, kernel, anchor=anchor)
    res = reErose(res, img, kernel, anchor=anchor)
    return res
