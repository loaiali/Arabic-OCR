from random import randint
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, random
from Preprocessing import textSkewCorrection, binarize
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy import stats


SCALE_PERCENT = 400  # percent of original s
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


def getMaxTransitions(lineIm,baseLine):
    maxTransition=0
    maxTransitionIndex = 0

    for i in range(0, baseLine-1):
        currentTransition=0
        flag=0
        #binary = cv2.line(lineIm, (0, i),
        #                      (lineIm.shape[1]-1, i), 255, 1)
        for j in range(0,lineIm.shape[1]):
            if lineIm[i,j]==1 and flag==0:
                currentTransition+=1
            if lineIm[i,j]!=1 and flag==1:
                flag=0
            
            #binary = cv2.line(lineIm, (j, i),
            #                     (j ,0), 255, 1)

            #cv2.imshow("LIne", binary)
            #cv2.waitKey(0)
                                  
        print(i, currentTransition)

        if currentTransition>=maxTransition:
            maxTransition=currentTransition
            maxTransitionIndex=i
    #cv2.imshow("LIne", lineIm)
    #cv2.waitKey(0)
            
    return maxTransitionIndex

def baseLineAndMaxLineDetection(lineIm):

    cv2.imshow("LIne", lineIm)
    cv2.waitKey(0)

    copp=lineIm.copy()
    copp[lineIm==255]=0
    copp[lineIm == 0] = 1


    #skeleton = skeletonize(copp)

    #skeleton = skeleton.astype(np.uint8)

    #skeleton[skeleton==1]=255

    #cv2.imshow("skeleton", skeleton)
    #cv2.waitKey(0)

    #skeleton[skeleton == 255] = 1

    #kernel = np.ones((3, 3), np.uint8)

    #img_dilation = cv2.erode(copp, kernel, iterations=1) # ;

    #img_dilation[img_dilation == 1] = 255

    #cv2.imshow("original", lineIm)
    #cv2.waitKey(0)

    #cv2.imshow("dialated", img_dilation)
    #cv2.waitKey(0)


    his = np.sum(copp, axis=1)

    maxx=np.argmax(his)

    maxT = getMaxTransitions(copp, maxx)


    # for print purooses ::::
    copp[copp == 0] = 255
    copp[copp == 1] = 0

    binary = cv2.line(copp, (0, maxx),
                      (copp.shape[1]-1, maxx), 0, 1)
    binary = cv2.line(copp, (0, maxT),
                      (copp.shape[1]-1, maxT), 0, 1)
    cv2.imshow("Imgage wih base and maxtransition Line", binary)
    cv2.waitKey(0)
    
    return maxx,maxT

def getMidCut(mfv,hisLine,start,midIndex,end,sr):

    print ("start ",start,midIndex,end)
    zeros=np.where(hisLine==0)[0]
    zeros = zeros[ zeros <= start]
    zeros = zeros[ zeros >= end]
    #print("zeros",zeros)
    if len(zeros)>0:
        #print("con",1)
        return min(zeros, key=lambda x: abs(x-midIndex))
    elif hisLine[midIndex] == mfv:
        #print("con",2)
        return midIndex
    
    lessThan = np.where(hisLine<=mfv)[0]
    lessThan = lessThan[lessThan <= start]
    lessThan = lessThan[lessThan >= end]
    lessThan2 = lessThan[lessThan <= midIndex]
    lessThan3 = lessThan[lessThan >= midIndex]


    #print("lessThan",lessThan)
    #print("lessThan2", lessThan2)
    #print("lessThan3", lessThan3)



    if len(lessThan2) > 1:
        return min(lessThan2, key=lambda x: abs(x-midIndex))
    if len(lessThan3) > 1 :
        return min(lessThan3, key=lambda x: abs(x-midIndex))
    else:
        #print("cond", 3)
        return midIndex

    
    


def cutPointIdentification(wordImage,MaxTransition):


    wordImage2 = wordImage.copy()
    wordImage2[wordImage == 255] = 1    # character
    wordImage2[wordImage == 0]   = 0    # background

    startIndex = wordImage2.shape[1]
    for i in reversed(range(0, wordImage2.shape[1])):
        if wordImage2[MaxTransition,i]==1 :
            startIndex=i+1
            break

    hisLine = np.sum(wordImage2, axis=0)
    print(hisLine)

    mfv=stats.mode(hisLine)[0][0]

    print("MFV =" , mfv)

    flag=0
    srList=[]
    sr={}
    #print("indie")
    count=0 
    for i in reversed(range(1, startIndex)):
        print("Pixel at max index i = ",i)
        if wordImage2[MaxTransition,i ]==1 and flag==1:
            #print("start")
            sr['end'] = i
            flag=0
            mid = getMidCut(mfv, hisLine, sr["start"], int((sr["start"]+sr["end"])/2), sr["end"], sr)
            count += 1

            print("cut point choosed ", mid)
            sr["mid"] = mid
            if True:
                print("special ", sr["start"], sr["end"])

                #binary = cv2.line(wordImage, (sr["start"], 0),
                #                  (sr["start"], wordImage.shape[1]-1),  255, 1)

                #binary = cv2.line(wordImage, (sr["end"], 0),
                #                             (sr["end"], wordImage.shape[1]-1), 255, 1)

                #binary = cv2.line(wordImage, (sr["mid"], 0),
                #                  (sr["mid"], wordImage.shape[1]-1), 128, 1)
                #cv2.circle(wordImage, (MaxTransition,mid),2,,-1)
            srList.append(sr)
        elif wordImage2[MaxTransition, i] != 1 and flag ==0:
            sr = {}
            sr["start"] = i
            flag = 1

        
    #binary = cv2.line(wordImage, (0, MaxTransition),
    #                  (wordImage.shape[1]-1, MaxTransition), 150, 1)

    return srList


def noConnectedPath(wordImage,baseLine,start,end):

    imageSub=wordImage[baseLine,end+1:start]
    for i in range(0, len(imageSub)):
        if imageSub[i]==0:
            return True

    return False




def seperationFiltrations(wordImage,srl, baseLine,maxTransition):
    wordImage2 = wordImage.copy()
    wordImage2[wordImage == 255] = 1    # character
    wordImage2[wordImage == 0]   = 0    # background


    hisLine = np.sum(wordImage2, axis=0)
    #print(hisLine)
    mfv = stats.mode(hisLine)[0][0]

    validSeperations=[]

    for sr in srl:
        start = sr["start"]
        end = sr["end"]

        if hisLine[sr["mid"]]==0:
            validSeperations.append(sr)
        elif noConnectedPath(wordImage2,baseLine,start,end):
            validSeperations.append(sr)
        elif hasHole(wordImage2,start,end):
        elif 
            pass



    return validSeperations


def underBaseLine(wordImage2 ,start, end,baseLine):

    hisLine = np.sum(wordImage2[:,start:end], axis=1) #horizontal histogram

    if noConnectedPath

    sumOveBbaseLine = np.sum(hisLine[0:baseLine])
    sumUnderBaseLine = np.sum(hisLine[baseLine+1:])






    pass

def printCut(img,srl):
    for sr in srl:
        binary = cv2.line(img, (sr["mid"], 0),
                      (sr["mid"], img.shape[1]-1), 128, 1)

    
    

def removeDotsAndHamza(img):
    img_white = img.copy()

    #img_white[:, :] = 255-img_white[:, :]
    #cv2.imshow("word wihtout dots", img_white)
    #cv2.waitKey(0)
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(
        img_white[:, :],)
    maxSize = -1
    maxLabel = -1
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        #print(label)
        if size > maxSize:
            #img_white[y:y+h, x:x+w] = 0
            maxSize = size
            maxLabel = label
            #print(maxSize)
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        if label != maxLabel:
            img_white[y:y+h, x:x+w] = 0

    #img_white[:, :] = 255-img_white[:, :]
    #print(labelimg)

    #cv2.imshow("word wihtout ", img_white)
    #cv2.waitKey(0)
    return img_white


def hasHole(wordImage,start,end):
    #print("befre remove")
    wordImage=removeDotsAndHamza(wordImage)
    #print ("after remove")
    _, contours, _ = cv2.findContours(wordImage[:, end:start].copy(),
                     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)

    cv2.drawContours(wordImage, contours, -1, 255, 3)
    
    return len(contours) > 1






def wordSegmentation(binary):


    binary2 = binary.copy()
    binary2[binary == 255] = 0
    binary2[binary == 0] = 1

    #print(np.sum(binary2, axis=1))

    # histogrma   axis = 1 for line sementation
    sol = segmentation(np.sum(binary2, axis=1), 1)
    wordList=[]
    for line in sol:
        begin, end = line

        
        #print(begin, end)
        #binary = cv2.line(binary, (0, begin),
        #                (binary.shape[1]-1, begin), 0, 1)
        #binary = cv2.line(binary, (0, end),
        #                (binary.shape[1]-1, end), 0, 1)

        width = int(binary2[begin:end][:].shape[1] * SCALE_PERCENT / 100)
        height = int(binary2[begin:end][:].shape[0] * 100 / 100)
        dim = (width, height)
        #print((width, height, binary2[begin:end]
        #    [:].shape[1], binary2[begin:end][:].shape[0]))
        # resize image
        binary3 = binary2[begin:end][:].copy()

        binary3[binary3 == 0] = 255
        binary3[binary3 == 1] = 0
        resized = cv2.resize(binary3, dim, interpolation=cv2.INTER_AREA)
        resizedcopy = resized.copy()

        #cv2.imshow("vertical lines", resized)
        #cv2.waitKey(0)
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
            #cv2.imshow("word segmented", resizedcopy[:,begin2:end2])
            #cv2.waitKey(0)
            begin2 = round(begin2*(100.0/SCALE_PERCENT))
            end2 = round(end2*(100.0/SCALE_PERCENT))
            subwords=[]



            
            baseLine,maxLine = baseLineAndMaxLineDetection(binary[begin:end,begin2:end2])
            binary[begin:end, begin2:end2] = 255 - binary[begin:end, begin2:end2]

            srl=cutPointIdentification(binary[begin:end, begin2:end2], maxLine)

            srl=seperationFiltrations(
                    binary[begin:end, begin2:end2], srl, baseLine, maxLine)
            
            #printCut(binary[begin:end, begin2:end2], srl)


            width = int(binary[begin:end, begin2:end2].shape[1]
                        * SCALE_PERCENT / 100)
            height = int(
                binary[begin:end, begin2:end2].shape[0] * SCALE_PERCENT / 100)
            dim = (width, height)
            resized = cv2.resize(
                binary[begin:end, begin2:end2], dim, interpolation=cv2.INTER_AREA)

            cv2.imshow("end", resized)
            cv2.waitKey(0)



            
            
            #binary = cv2.line(binary, (begin2, begin),
            #                      (begin2, end), 180,1)
            #binary = cv2.line(binary, (end2, begin),
            #                      (end2, end), 180, 1)
            word = {'rows': (begin, end), 'columns': (begin2, end2), 'subwords': subwords,}
            lineList.insert(0, word)
        wordList=wordList+lineList


    #imgplot = plt.imshow(binary)
    #plt.show()
    #cv2.imshow("word segmented", binary)
    #cv2.waitKey(0)

    return wordList








def main():

    for i in range (0,1):

        file=random.choice(os.listdir("scanned\\"))
        img = cv2.imread("capr2.png")#"scanned\\"+file)

        thre = binarize(img)
        rotated = textSkewCorrection(thre)
        wordList=wordSegmentation(rotated)

        for word in wordList:
            c1,c2 = word['rows']
            r1_offset,r1_offset2=word['columns']
            vindex=word['vindex']
            if len(word['subwords'])==0:
                cv2.imshow("sub word ", rotated[c1:c2, r1_offset:r1_offset2])
                cv2.waitKey(0)
            else:
                for subword in word['subwords']:
                    r1, r2 = subword 
                    cv2.imshow("sub word ", rotated[c1:c2, r1_offset+r1:r1_offset+r2])
                    cv2.waitKey(0)

                
                

main()








