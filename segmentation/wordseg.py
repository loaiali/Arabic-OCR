import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, random
from Preprocessing import textSkewCorrection, binarize
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy import stats


SCALE_PERCENT = 200  # percent of original 
CHAR_DI=100


def segmentation(histogram,thresold):
    bline = True
    sol = [] 
    i=0
    while i < len(histogram)-1 :
        #print(i)
        while i < len(histogram)-1 and histogram[i+1]==0 and histogram[i] == 0:
            i+=1
            #print(i)
        start=i
        i=i+1
        while i < len(histogram)-1:
            #print(i)
            while i<len(histogram)-1 and  histogram[i+1]!=0 and histogram[i] != 0:
                #print(i)
                i+=1
                
            index =i
            while i < len(histogram)-1 and histogram[i+1] == 0:
                #print(i)
                i=i+1

            if (i-index > thresold):
                i+=1
                end = index
                sol.append((start,end))
                break
            i=i+1
    return sol


def wordSegmentation(binary):


    binary2 = binary.copy()
    binary2[binary == 255] = 0
    binary2[binary == 0] = 1

    sol = segmentation(np.sum(binary2, axis=1), 1)
    wordList=[]
    for line in sol:
        begin, end = line

        

        width = int(binary2[begin:end][:].shape[1] * SCALE_PERCENT / 100)
        height = int(binary2[begin:end][:].shape[0] * 100 / 100)
        dim = (width, height)
   
        binary3 = binary2[begin:end][:].copy()

        binary3[binary3 == 0] = 255
        binary3[binary3 == 1] = 0
        resized = cv2.resize(binary3, dim, interpolation=cv2.INTER_AREA)
        resizedcopy = resized.copy()


        resized[resizedcopy > 128] = 0
        resized[resizedcopy < 128] = 1



        ## words segminations
        sol2 = segmentation(np.sum(resized, axis=0), 5)
        lineList=[]
        for hor in sol2:
            begin2, end2 = hor
            #print(begin2,end2)
            sol3 = segmentation(
                np.sum(resized[:, begin2:end2], axis=0), 0)
            begin2 = round(begin2*(100.0/SCALE_PERCENT))
            end2 = round(end2*(100.0/SCALE_PERCENT))
            subwords=[]


            for subword in sol3:
                begin3,end3=subword
                begin3 = round(begin3*(100.0/SCALE_PERCENT))
                end3 = round(end3*(100.0/SCALE_PERCENT))

                subwords.append((begin3, end3))
                
            

            word = {'rows': (begin, end), 'columns': (begin2, end2), 'subwords': subwords,}
            lineList.insert(0, word)
        wordList=wordList+lineList


    return wordList








def main():

    for i in range (0,10):

        file=random.choice(os.listdir("scanned\\"))
        img = cv2.imread("scanned\\"+file)

        thre = binarize(img)
        rotated = textSkewCorrection(thre)
        wordList=wordSegmentation(rotated)

        for word in wordList:
            c1,c2 = word['rows']
            r1_offset,r1_offset2=word['columns']
            if len(word['subwords'])==0:
                cv2.imshow("sub word ", rotated[c1:c2, r1_offset:r1_offset2])
                cv2.waitKey(0)
            else:
                for subword in word['subwords']:
                    r1, r2 = subword 
                    cv2.imshow("sub word ", rotated[c1:c2, r1_offset+r1:r1_offset+r2])
                    cv2.waitKey(0)

                
                









