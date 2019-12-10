from random import randint
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, random
from Preprocessing import textSkewCorrection, binarize
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy import stats
import heapq

SCALE_PERCENT = 400  
CHAR_DI=100

def showScaled(img, text, scale):
    width = int(img.shape[1]
                * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow(text, resized)
    cv2.waitKey(0)


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
    #print(hisLine)

    mfv=stats.mode(hisLine)[0][0]

    #print("MFV =" , mfv)

    flag=0
    srList=[]
    sr={}
    #print("indie")
    count=0 
    for i in reversed(range(1, startIndex)):
        #print("Pixel at max index i = ",i)
        if wordImage2[MaxTransition,i ]==1 and flag==1:
            #print("start")
            sr['end'] = i
            flag=0
            mid = getMidCut(mfv, hisLine, sr["start"], int((sr["start"]+sr["end"])/2), sr["end"], sr)
            count += 1

            #print("cut point choosed ", mid)
            sr["mid"] = mid
            if count ==6:
                print("special",sr["start"],sr["end"])
                #print("special ", sr["start"], sr["end"])
                pass
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


def noConnectedBaseLine(wordImage,baseLine,start,end):

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
    i=0
    
    while i < len(srl):
        start = srl[i]["start"]
        end = srl[i]["end"]
        print("*******************************cut Number = ",i,"  start", "end = " ,start,end, "mfv = ", mfv)

        serp=[]
        seg=[]
        segn=[]
        segnn=[]
        if  i ==0 or i ==len(srl)-1:
            serp.append(srl[i]["start"])
            serp.append(srl[i]["end"])
        else:
            serp.append(srl[i-1]["mid"])
            serp.append(srl[i+1]["mid"])
        
        if i!=len(srl)-1:
            seg.append(srl[i]["mid"])
            seg.append(srl[i+1]["mid"])

        if i < len(srl)-2 :
            segn.append(srl[i+1]["mid"])
            segn.append(srl[i+2]["mid"])
        if i < len(srl)-3:
            segnn.append(srl[i+2]["mid"])
            segnn.append(srl[i+3]["mid"])



        if hisLine[srl[i]["mid"]]==0:
            print("cond 1")
            validSeperations.append(srl[i])
            i+=1
        elif hasHole(wordImage2,serp[0],serp[1]):                # unfortunately remove some correct cuts
            print("cond 2")
            i+=1
            continue
        elif  noConnectedBaseLine(wordImage2, baseLine, start, end):
            print("cond 3")
            if (underBaseLine(wordImage2, start, end, baseLine)):
                i+=1
                continue
            elif hisLine[srl[i]["mid"]] < mfv :
                validSeperations.append(srl[i])
                i += 1
            else :
                i+=1
                continue
        elif srl[i] == srl[-1] and checkSegmentLength(wordImage2, start, end, baseLine):
            i+=1
            continue
        elif ((len(seg) != 0) and not isStroke(wordImage2, seg[0], seg[1], baseLine, mfv)) or ((len(seg)!=0) and (len(segn)!=0) and not hasHole(wordImage2, seg[0], seg[1]) and not hasHole(wordImage2, segn[0], segn[1])and  hasHole(wordImage2, seg[0], segn[1])):
            print("SEG not stroke ")
            if  (i <len(srl)-1) and noConnectedBaseLine(wordImage2, baseLine, srl[i+1]["start"], srl[i+1]["end"]) and hisLine[srl[i+1]["mid"]] <= mfv:
                print("SEG no stroke and noConnectedBaseLine")
                i+=1
                continue
            else:
                print("SEG no stroke and ConnectedBaseLine ---> accepted i=i+1")
                validSeperations.append(srl[i])
                i += 1
        elif (len(seg) != 0)and isStroke(wordImage2, seg[0], seg[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, seg[0], seg[1]):
                print("SEG Is stroke and Dots ----> accepted i=i+1")
                validSeperations.append(srl[i])
                i=i+1
        elif (len(seg) != 0)and isStroke(wordImage2, seg[0], seg[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, seg[0], seg[1]):
            print("SEG Is stroke and No Dots")
            if (len(segn) != 0)and isStroke(wordImage2, segn[0], segn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segn[0], segn[1]):
                print("SEGN Is stroke and No Dots  ---> accpeted i=i+3")
                validSeperations.append(srl[i])
                i=i+3
                continue
            if (len(segnn) != 0) and isStroke(wordImage2, segn[0], segn[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, segn[0], segn[1]) and isStroke(wordImage2, segnn[0], segnn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segnn[0], segnn[1]):
                print( "SEGNN Is stroke and Dots  and SEGNN is stroke and no dots ---> accpeted i=i+3")
                validSeperations.append(srl[i])
                i = i+3
                continue
            if (len(segn) != 0)and ( not isStroke(wordImage2, segn[0], segn[1], baseLine, mfv) or (isStroke(wordImage2, segn[0], segn[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, segn[0], segn[1]))):
                print("SEGN Is not  stroke  or segn is stroke with Dots ")
                i=i+1
                continue
            i=i+1
        else :
            print("cond else")
            validSeperations.append(srl[i])
            i += 1



    return validSeperations


def dotBelowOrAbove(img, start, end,thres=12):
    _, contours, _ = cv2.findContours(img[:, end:start].copy(),
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours) > 1


def isStroke(wordImage2, start, end, baseLine,mfv,alfLength=13,error=5):
    img = wordImage2[:,end:start].copy()

    print("*******is Stroke function results****")
    labelnum, _, _, _ = cv2.connectedComponentsWithStats(img)
    if labelnum >2:
        return False
    print("is single connected conponens")

    if  underBaseLine(wordImage2[:, end:start], start, end,baseLine):
        return False
    print("Not under base Line")


    hisLine = np.sum(wordImage2[:, 0:start], axis=1)  # اhoriontal histogram


    mfvHorizontal = stats.mode(hisLine)[0][0]

    print(mfvHorizontal,mfv)

    if (abs(int(mfvHorizontal)- int(mfv)) >2 ):
        return False
    print("3ard el stroke =  base Line",  (abs(int(mfvHorizontal) - int(mfv))))

    #if i>0 and i <len(srl)-1:
    #    if hasHole(wordImage2, srl[i-1]["mid"], srl[i+1]["mid"]):
    #        return False

    if hasHole(wordImage2, start, end):
        return False

    print("No hole")

    biggest =biggestConnectedComponent(wordImage2[:, end:start])
    h=calculateHeight(biggest)
    print("h equal", h)

    if (h>alfLength-error ):
        return False

    print("Stroke lenth is good")


    
    


    return True


def checkSegmentLength(wordImage2, start, end, baseLine,thres=9):
    hisLine = np.sum(wordImage2[:,0:start], axis=1) # vertical histogram

    indx = np.where(hisLine > 0)[0]

    if len(indx) <=0 :
        return False
    mini = indx[0]
    maxi = indx[-1]
    
    if abs((maxi-mini)) <=thres :
        return True
    
    return False



def underBaseLine(wordImage2 ,start, end,baseLine):

    hisLine = np.sum(wordImage2[:,end:start], axis=1) #horizontal histogram

    sumOveBbaseLine = np.sum(hisLine[0:baseLine])
    sumUnderBaseLine = np.sum(hisLine[baseLine+1:])

    if sumUnderBaseLine > sumOveBbaseLine :
        return True
    
    return False




def printCut(img,srl):
    for sr in srl:
        binary = cv2.line(img, (sr["mid"], 0),
                      (sr["mid"], img.shape[1]-1), 128, 1)

    
    

def removeDotsAndHamza(img,thres=12):
    img_white = img.copy()

    #img_white[:, :] = 255-img_white[:, :]
    #cv2.imshow("word wihtout dots", img_white)
    #cv2.waitKey(0)
    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(
        img_white[:, :],)
    maxLabel = []
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        #print(size)
        if size > thres:
            maxLabel.append(label)
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        if label not in  maxLabel:
            img_white[y:y+h, x:x+w] = 0


    return img_white


def biggestConnectedComponent(img ):
    img_white = img.copy()

    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(
        img_white[:, :],)
    maxLabel = -1
    maxa=-1
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        #print(size)
        if size > maxa:
            maxLabel=label
            maxa=size
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        if label != maxLabel:
            img_white[y:y+h, x:x+w] = 0
    return img_white

def calculateHeight(img):
    hisLine = np.sum(img, axis=1)
    indx = np.sort(np.where(hisLine > 0)[0])
    if len(indx) <= 0:
        return -1
    mini = indx[0]
    maxi = indx[-1]
    return abs((maxi-mini)) 
    






def hasHole(wordImage,start,end):
    #print("befre remove")

    if start ==-1 and end ==-1 :
        return False
    wordImage=removeDotsAndHamza(wordImage)
    #print (wordImage[:,end:start])

    #print ("after remove")
    _, contours, _ = cv2.findContours(wordImage[:, end+1:start].copy(),
                     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(contours)

    cv2.drawContours(wordImage, contours, -1, 255, 3)
    
    return len(contours) > 1



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
        #print((width, height, binary2[begin:end]
        #    [:].shape[1], binary2[begin:end][:].shape[0]))
        # resize image
        binary3 = binary2[begin:end][:].copy()

        binary3[binary3 == 0] = 255
        binary3[binary3 == 1] = 0
        resized = cv2.resize(binary3, dim, interpolation=cv2.INTER_AREA)
        resizedcopy = resized.copy()

        resized[resizedcopy > 128] = 0
        resized[resizedcopy < 128] = 1

        ## words segminations
        sol2 = segmentation(np.sum(resized, axis=0), 8)
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

            imgWithAllCuts=binary[begin:end, begin2:end2].copy()
            printCut(imgWithAllCuts, srl)

            srl=seperationFiltrations(
                    binary[begin:end, begin2:end2], srl, baseLine, maxLine)
            
            printCut(binary[begin:end, begin2:end2], srl)

            showScaled(imgWithAllCuts,
                                   "image with All cuts", 400)
            showScaled(binary[begin:end, begin2:end2],
                       "image with correct filtrarions", 400)



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







