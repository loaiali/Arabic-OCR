import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt

def binarize(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def segmentLine(img):
    pass




def textSkewCorrection(thresh):
    #get all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    #get the angle of minAreaRect
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
    	angle = -(90 + angle)
    else:
        angle=-angle
    
    #rotation
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    #rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # show the output image
    #print("[INFO] angle: {:.3f}".format(angle))
    #cv2.imwrite("rotated.png",rotated)
    #cv2.imshow("Input", thresh)
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    rotated = cv2.threshold(
        rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  
    return cv2.bitwise_not(rotated)


#textSkewCorrection(image)
def preprocessImageFromPath(img):
    img = cv2.imread(img)
    return img
    return preprocessImage(img)

def preprocessImage(img):
    return img
    binarizedImage=binarize(img)
    textSkewCorrectedImg=textSkewCorrection(binarizedImage)
    return textSkewCorrectedImg
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

