# -*- coding: utf-8 -*-
# Reference: https://sunhwee.com/

import cv2
import math
import numpy as np

def gaussKernel(gaussKernelSize, sigma):
    gaussKernel = np.zeros((gaussKernelSize,gaussKernelSize),np.float32)
    for i in range (gaussKernelSize):
        for j in range (gaussKernelSize):
            norm = math.pow(i-1,2)+pow(j-1,2)
            gaussKernel[i,j] = math.exp(-norm/(2*math.pow(sigma,2)))   # 求高斯卷积
    sum = np.sum(gaussKernel)   # 求和
    kernel = gaussKernel/sum   # 归一化
    return kernel

def gaussian(photoPath, gaussKernelSize, sigma):
    if len(photoPath)>0:
        image = cv2.imread(photoPath)

        kernel = np.array(gaussKernel(gaussKernelSize, sigma), dtype=float)
        tmpImage = cv2.filter2D(image, cv2.CV_16S, kernel)
        Gauss = cv2.convertScaleAbs(tmpImage)

        # 保存图片
        outImageName = photoPath
        outImageName = 'gauss' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Gauss)
        return Gauss
    else:
        return None

def meanKernel(meanKernelSize):
    meanKernel = np.zeros((meanKernelSize,meanKernelSize),np.float32)
    for i in range(meanKernelSize):
        for j in range(meanKernelSize):
            meanKernel[i,j] = 1.0
    sum = np.sum(meanKernel)
    kernel = meanKernel/sum
    return kernel

def mean(photoPath, meanKernelSize):
    if len(photoPath)>0:
        image = cv2.imread(photoPath)

        kernel = np.array(meanKernel(meanKernelSize), dtype=float)
        tmpImage = cv2.filter2D(image, cv2.CV_16S, kernel)
        Mean = cv2.convertScaleAbs(tmpImage)

        # 保存图片
        outImageName = photoPath
        outImageName = 'mean' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Mean)
        return Mean
    else:
        return None

def median(photoPath, medianKernelSize):
    edgeW = medianKernelSize//2
    if len(photoPath)>0:
        image = cv2.imread(photoPath)
        height = image.shape[0]
        weight = image.shape[1]

        blueImg, greenImg, redImg = cv2.split(image)

        BOutImage = np.zeros((height, weight), np.uint8)
        GOutImage = np.zeros((height, weight), np.uint8)
        ROutImage = np.zeros((height, weight), np.uint8)
        for i in range(0, height):
            for j in range(0, weight):
                if (i>=edgeW and i<height-edgeW
                    and j>=edgeW and j<weight-edgeW):
                    BTmp = np.zeros(medianKernelSize**2, np.uint8)
                    GTmp = np.zeros(medianKernelSize**2, np.uint8)
                    RTmp = np.zeros(medianKernelSize**2, np.uint8)
                    s = 0
                    for m in range(-edgeW, edgeW+1):
                        for n in range(-edgeW, edgeW+1):
                            BTmp[s] = blueImg[i+m,j+n]
                            GTmp[s] = greenImg[i+m,j+n]
                            RTmp[s] = redImg[i+m,j+n]
                            s+=1
                    BTmp = np.sort(BTmp)
                    GTmp = np.sort(GTmp)
                    RTmp = np.sort(RTmp)
                    BOutImage[i,j] = BTmp[medianKernelSize**2//2]
                    GOutImage[i,j] = GTmp[medianKernelSize**2//2]
                    ROutImage[i,j] = RTmp[medianKernelSize**2//2]
                else:
                    BOutImage[i,j] = blueImg[i,j]
                    GOutImage[i,j] = greenImg[i,j]
                    ROutImage[i,j] = redImg[i,j]
        Median = cv2.merge([BOutImage,GOutImage,ROutImage])

        # 保存图片
        outImageName = photoPath
        outImageName = 'median' + outImageName.replace('/','.')
        cv2.imwrite(outImageName, Median)
        return Median
    else:
        return None
        
