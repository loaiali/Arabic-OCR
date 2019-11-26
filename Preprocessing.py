import numpy as np
import argparse
import cv2


def textSkewCorrection():
    image=cv2.imread("capr496.png")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray=cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    

    #get all white pixels
    coords = np.column_stack(np.where(thresh > 0))
    
    #get the angle of minAreaRect
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
    	angle = -(90 + angle)
    else:
        angle=-angle
    
    #rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    #rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)    
    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    cv2.imwrite("rotated.png",rotated)
    cv2.imshow("Input", image)
    cv2.imshow("Rotated", rotated)
    cv2.waitKey(0)     



if __name__=="__main__":
    textSkewCorrection()