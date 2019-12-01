import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt

import cv2
import math
import tkinter.filedialog
import tkinter.messagebox 

from PIL import Image
from PIL import ImageTk

import operation as op
import edgeDetect as ed

def showImages(row, col, num, titles, images, type='gray'):
    for i in np.arange(num):
       plt.subplot(row,col,i+1)
       plt.imshow(images[i], type)
       plt.title(titles[i])
       plt.xticks([])
       plt.yticks([])
    plt.show()
    return

def saveImage(name, image):
    global photoPath
    outImageName = photoPath
    outImageName = name + outImageName.replace('/','.')
    cv2.imwrite(outImageName, image)
    return

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

def genSE():
    global structureElement, seSize
    kernel = None
    if len(structureElement)==0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, seSize)
    else:
        kernel = structureElement
    return kernel

def edgeDetStand():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        kernel = genSE()
        res = ed.morED(binary, kernel, anchor=anchor, flag=ed.STANDARD)

        saveImage('EDS', res)

        titles = ['binary', 'result']
        images = [binary, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def edgeDetExter():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        kernel = genSE()
        res = ed.morED(binary, kernel, anchor=anchor, flag=ed.EXTERNAL)

        saveImage('EDE', res)

        titles = ['binary', 'result']
        images = [binary, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def edgeDetInter():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        kernel = genSE()
        res = ed.morED(binary, kernel, anchor=anchor, flag=ed.INTERNAL)

        saveImage('EDI', res)

        titles = ['binary', 'result']
        images = [binary, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def gradientStand():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = genSE()
        res = ed.gradient(img, kernel, anchor=anchor, flag=ed.STANDARD)

        saveImage('GS', res)

        titles = ['gray', 'result']
        images = [img, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def gradientExter():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = genSE()
        res = ed.gradient(img, kernel, anchor=anchor, flag=ed.EXTERNAL)

        saveImage('GE', res)

        titles = ['gray', 'result']
        images = [img, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def gradientInter():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = genSE()
        res = ed.gradient(img, kernel, anchor=anchor, flag=ed.INTERNAL)

        saveImage('GI', res)

        titles = ['gray', 'result']
        images = [img, res]
        showImages(1,2,2,titles,images)
        return
    else:
        return

def condDilate():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        kernel = genSE()
        marker = cv2.erode(binary, kernel, anchor=anchor, iterations=10)
        res = op.condDilate(marker, binary, kernel, anchor=anchor)

        saveImage('CondDilate', res)

        titles = ['marker', 'mask', 'result']  
        images = [marker, binary, res]  
        showImages(1,3,3,titles,images)
        return
    else:
        return

def opOpenRe():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = genSE()
        res = op.openRe(img, kernel, anchor=anchor)

        saveImage('OBR', res)

        titles = ['gray', 'result']   
        images = [img, res] 
        showImages(1,2,2,titles,images)
        return
    else:
        return

def opCloseRe():
    global photoPath, structureElement, seSize, anchor
    if len(photoPath)>0:
        img = cv2.imread(photoPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = genSE()
        res = op.closeRe(img, kernel, anchor=anchor)

        saveImage('CBR', res)

        titles = ['gray', 'result']   
        images = [img, res] 
        showImages(1,2,2,titles,images)
        return
    else:
        return

def testDigit(content):
    if (content.isdigit() or content == ''):
        return True
    elif content[0]=='-' and content[1:].isdigit():
        return True
    else:
        return False

def setSESize():
    global window, seSize

    settingWindow = tk.Toplevel(window)
    settingWindow.geometry('300x100')
    settingWindow.title('SE Size')

    test_cmd = settingWindow.register(testDigit)

    heightString = tk.StringVar(master=settingWindow, value=seSize[0])
    tk.Label(settingWindow, text='SE height: ').grid(row=1, column=0)
    entryHeight = tk.Entry(
        settingWindow, 
        textvariable=heightString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryHeight.grid(row=1, column=1)

    weightString = tk.StringVar(master=settingWindow, value=seSize[1])
    tk.Label(settingWindow, text='SE Weight: ').grid(row=2, column=0)
    entryWeight = tk.Entry(
        settingWindow, 
        textvariable=weightString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryWeight.grid(row=2, column=1)

    def save():
        global seSize
        newSize = (int(heightString.get()),int(weightString.get()))
        seSize = newSize
        settingWindow.destroy()
        return

    button = tk.Button(settingWindow, text='Save', command=save)
    button.grid(row=3, column=0, columnspan=2)
    return 

def setAnchor():
    global window, anchor

    settingWindow = tk.Toplevel(window)
    settingWindow.geometry('300x100')
    settingWindow.title('Anchor')

    test_cmd = settingWindow.register(testDigit)

    heightString = tk.StringVar(master=settingWindow, value=anchor[0])
    tk.Label(settingWindow, text='Anchor x: ').grid(row=1, column=0)
    entryHeight = tk.Entry(
        settingWindow, 
        textvariable=heightString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryHeight.grid(row=1, column=1)

    weightString = tk.StringVar(master=settingWindow, value=anchor[1])
    tk.Label(settingWindow, text='Anchor y: ').grid(row=2, column=0)
    entryWeight = tk.Entry(
        settingWindow, 
        textvariable=weightString,
        validate='key',
        validatecommand=(test_cmd, '%P')
    )
    entryWeight.grid(row=2, column=1)

    def save():
        global anchor
        newAnchor = (int(heightString.get()),int(weightString.get()))
        anchor = newAnchor
        settingWindow.destroy()
        return

    button = tk.Button(settingWindow, text='Save', command=save)
    button.grid(row=3, column=0, columnspan=2)
    return 

def setSE():
    global window, seSize, structureElement

    settingWindow = tk.Toplevel(window)
    # settingWindow.geometry('600x600')
    settingWindow.title('SE Size')

    test_cmd = settingWindow.register(testDigit)
    rowNum = int(seSize[0])
    colNum = int(seSize[1])

    tmp = []
    for i in range(0, rowNum):
        row = []
        for j in range(0, colNum):
            String = None
            if len(structureElement)==0:
                String = tk.StringVar(master=settingWindow, value='')
            else:
                String = tk.StringVar(master=settingWindow, value=structureElement[i][j])
            entry = tk.Entry(
                settingWindow, 
                textvariable=String,
                validate='key',
                validatecommand=(test_cmd, '%P')
            )
            entry.grid(row=i, column=j)
            item = (String, entry)
            row.append(item)
        tmp.append(row)

    se = np.zeros(seSize, np.uint8)
    def save():
        global structureElement, seSize
        for i in range(0, rowNum):
            for j in range(0, colNum):
                item = tmp[i][j]
                string = item[0].get()
                num = 0
                if len(string)>0:
                    num = int(string)
                se[i,j] = int(num)
        structureElement = se
        settingWindow.destroy()
        return

    def clean():
        global structureElement, seSize
        for i in range(0, rowNum):
            for j in range(0, colNum):
                item = tmp[i][j]
                entry = item[1]
                entry.delete(0,tk.END)
        structureElement = []
        return
    
    buttonSave = tk.Button(settingWindow, text='Save', command=save)
    buttonSave.grid(row=rowNum,column=0,columnspan=colNum)
    buttonClean = tk.Button(settingWindow, text='Clean', command=clean)
    buttonClean.grid(row=rowNum+1,column=0,columnspan=colNum)
    return

def cleanSE():
    global structureElement
    if len(structureElement)>0:
        structureElement = []
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

seSize = (5, 5)
anchor = (-1, -1)
structureElement = []



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

submenuEd = tk.Menu(editmenu)
editmenu.add_cascade(label='Edge Detect', menu=submenuEd)
submenuEd.add_command(label='Standard', command=edgeDetStand)
submenuEd.add_command(label='External', command=edgeDetExter)
submenuEd.add_command(label='Internal', command=edgeDetInter)

submenuG = tk.Menu(editmenu)
editmenu.add_cascade(label='Morphological gradient', menu=submenuG)
submenuG.add_command(label='Standard', command=gradientStand)
submenuG.add_command(label='External', command=gradientExter)
submenuG.add_command(label='Internal', command=gradientInter)

editmenu.add_command(label='Conditional Dilate', command=condDilate)
editmenu.add_command(label='OBR', command=opOpenRe)
editmenu.add_command(label='CBR', command=opCloseRe)

# "Setting" menu
setmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='Setting', menu=setmenu)

setmenu.add_command(label='SE Size', command=setSESize)
setmenu.add_command(label='Anchor', command=setAnchor)
setmenu.add_command(label='Structure Element', command=setSE)
setmenu.add_command(label='Clean SE', command=cleanSE)

# "Help" menu
menubar.add_command(label='Help', command=helpCmd)

window.config(menu=menubar)

window.mainloop()