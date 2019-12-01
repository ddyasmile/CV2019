import cv2

STANDARD = 0
EXTERNAL = 1
INTERNAL = 2

def morED(img, kernel, anchor=(-1,-1), flag=STANDARD):
    eroded = cv2.erode(img, kernel, anchor=anchor)
    dilated = cv2.dilate(img, kernel, anchor=anchor)
    res = None
    if flag == STANDARD:
        res = cv2.subtract(dilated, eroded)
    elif flag == EXTERNAL:
        res = cv2.subtract(dilated, img)
    elif flag == INTERNAL:
        res = cv2.subtract(img, eroded)
    return res

def gradient(img, kernel, anchor=(-1,-1), flag=STANDARD):
    eroded = cv2.erode(img, kernel, anchor=anchor)
    dilated = cv2.dilate(img, kernel, anchor=anchor)
    res = None
    if flag == STANDARD:
        res = 0.5*cv2.subtract(dilated, eroded)
    elif flag == EXTERNAL:
        res = 0.5*cv2.subtract(dilated, img)
    elif flag == INTERNAL:
        res = 0.5*cv2.subtract(img, eroded)
    return res