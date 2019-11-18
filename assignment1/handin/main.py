# -*- coding: utf-8 -*-
# Reference: https://sunhwee.com/


import cv2
import math
import tkinter.filedialog
import tkinter.messagebox 
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import edgeDetect as myed
import filter as myfilter

from PIL import Image
from PIL import ImageTk


# file operation: open save
def fitSize(photoSize):
    global winSize
    xyRate = float(photoSize[0])/float(photoSize[1])
    yTmp = float(winSize[0])/xyRate
    xTmp = float(winSize[1])*xyRate

    if yTmp>winSize[1]:
        return (int(xTmp), winSize[1])
    else:
        return (winSize[0],int(yTmp))

def openPhoto():
    openPath = tkinter.filedialog.askopenfilename()
    if len(openPath)>0:
        global photoPath, panel
        photoPath = openPath
        image = cv2.imread(openPath)
        
        imageTmp = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        imageShow = ImageTk.PhotoImage(imageTmp.resize(fitSize(imageTmp.size)))

        if (panel is None):
            panel = tk.Label(image=imageShow)
            panel.image = imageShow
            panel.pack()
        else:
            panel.configure(image=imageShow)
            panel.image = imageShow

    return

# photo operation
# Roberts operator; Prewitt operator; Sobel operator;
# Gaussian filter, mean filter and Median filter
def robertsOp():
    global photoPath
    Roberts = myed.roberts(photoPath)
    if Roberts is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(Roberts)
    plt.show()
    return

def prewittOp():
    global photoPath
    Prewitt = myed.prewitt(photoPath)
    if Prewitt is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(Prewitt)
    plt.show()
    return

def sobelOp():
    global photoPath
    Sobel = myed.sobel(photoPath)
    if Sobel is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(Sobel)
    plt.show()
    return

def gaussianFilter():
    global photoPath
    global gaussKernelSize, sigma
    Gauss = myfilter.gaussian(photoPath, gaussKernelSize, sigma)
    if Gauss is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(cv2.cvtColor(Gauss, cv2.COLOR_BGR2RGB))
    plt.show()
    return

def meanFilter():
    global photoPath
    global meanKernelSize
    Mean = myfilter.mean(photoPath, meanKernelSize)
    if Mean is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(cv2.cvtColor(Mean, cv2.COLOR_BGR2RGB))
    plt.show()
    return

def medianFilter():
    global photoPath
    global medianKernelSize
    Median = myfilter.median(photoPath, medianKernelSize)
    if Median is None:
        return
    ## 显示结果
    plt.plot([])
    plt.imshow(cv2.cvtColor(Median, cv2.COLOR_BGR2RGB))
    plt.show()
    return

def changeSigma(param):
    global sigma
    sigma = float(param)
    return

def testDigit(content):
    if content.isdigit() or content == "":
        return True
    else:
        return False

def setting():
    global window
    global sigma, gaussKernelSize
    global meanKernelSize, medianKernelSize

    settingWindow = tk.Toplevel(window)
    settingWindow.geometry('300x200')
    settingWindow.title('Setting')

    test_cmd = settingWindow.register(testDigit)

    gaussString = tk.StringVar(master=settingWindow, value=gaussKernelSize)
    tk.Label(settingWindow, text='Gauss Kernel Size: ').grid(row=1, column=0)
    entryGuass = tk.Entry(
        settingWindow, 
        textvariable=gaussString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryGuass.grid(row=1, column=1)

    meanString = tk.StringVar(master=settingWindow, value=meanKernelSize)
    tk.Label(settingWindow, text='Mean Kernel Size: ').grid(row=2, column=0)
    entryMean = tk.Entry(
        settingWindow, 
        textvariable=meanString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryMean.grid(row=2, column=1)

    medianString = tk.StringVar(master=settingWindow, value=medianKernelSize)
    tk.Label(settingWindow, text='Median Kernel Size: ').grid(row=3, column=0)
    entryMedian = tk.Entry(
        settingWindow, 
        textvariable=medianString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryMedian.grid(row=3, column=1)

    sigmaScale = tk.Scale(
        settingWindow,
        label='sigma',
        from_=0.1, to=30.0,
        orient=tk.HORIZONTAL, 
        length=200,
        tickinterval=10, 
        resolution=0.01,
        command=changeSigma
    )
    sigmaScale.grid(row=4, column=0, columnspan=2, rowspan=2)

    def saveAllSet():
        global gaussKernelSize
        global meanKernelSize
        global medianKernelSize
        localGaussKernelSize = int(gaussString.get())
        localMeanKernelSize = int(meanString.get())
        localMedianKernelSize = int(medianString.get())
        if (localGaussKernelSize%2==0
            or localMeanKernelSize%2==0
            or localMedianKernelSize%2==0):
            tkinter.messagebox.showwarning(title='Warning', message='Odd needed! ')
            return
        gaussKernelSize = int(gaussString.get())
        meanKernelSize = int(meanString.get())
        medianKernelSize = int(medianString.get())
        settingWindow.destroy()
        return

    button = tk.Button(settingWindow, text='OK', command=saveAllSet)
    button.grid(row=6, column=0, columnspan=2)

    return

def helpCmd():
    global window
    helpFile = open('readme', 'r')
    helpString = helpFile.read()
    helpFile.close()

    helpWindow = tk.Toplevel(window)
    helpWindow.geometry('720x560')
    helpWindow.title('README')

    text = tk.Text(helpWindow, wrap=tk.WORD)
    text.insert(tk.INSERT, helpString)
    text.pack()
    return

window = tk.Tk()
window.title('Photo Shop Mini')
window.geometry('1080x720')

winSize = (1080,720)
photoPath = ''
panel = None

sigma = 1.0
gaussKernelSize = 3
meanKernelSize = 3
medianKernelSize = 3

# menu bar, has some funtions
menubar = tk.Menu(window)

# "File" menu
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Open', command=openPhoto)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

# "Edit" menu
editmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Edit', menu=editmenu)
editmenu.add_command(label='Robert', command=robertsOp)
editmenu.add_command(label='Prewitt', command=prewittOp)
editmenu.add_command(label='Sobel', command=sobelOp)
editmenu.add_command(label='Gaussian Filter', command=gaussianFilter)
editmenu.add_command(label='Mean Filter', command=meanFilter)
editmenu.add_command(label='Median Filter', command=medianFilter)

# "Setting" menu
menubar.add_command(label='Setting', command=setting)

# "Help" menu
menubar.add_command(label='Help', command=helpCmd)

window.config(menu=menubar)

window.mainloop()


