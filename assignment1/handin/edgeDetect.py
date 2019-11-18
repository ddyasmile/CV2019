# -*- coding: utf-8 -*-
# Reference: https://sunhwee.com/

import cv2
import numpy as np


def roberts(photoPath):
    if len(photoPath)>0:
        image = cv2.imread(photoPath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[-1,0],[0,1]], dtype=int)
        kernely = np.array([[0,-1],[1,0]], dtype=int)
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 保存图片
        outImageName = photoPath
        outImageName = 'robert' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Roberts)
        return Roberts
    else:
        return None

def prewitt(photoPath):
    if len(photoPath)>0:
        image = cv2.imread(photoPath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 保存图片
        outImageName = photoPath
        outImageName = 'prewitt' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Prewitt)
        return Prewitt
    else:
        return None

def sobel(photoPath):
    if len(photoPath)>0:
        image = cv2.imread(photoPath)
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        kernelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=int)
        kernely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=int)
        x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
        y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        # 保存图片
        outImageName = photoPath
        outImageName = 'sobel' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Sobel)
        return Sobel
    else:
        return None