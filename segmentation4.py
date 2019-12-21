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
    length = len(histogram)-1
    while i < length :
        #print(i)
        while i < length and histogram[i+1] == 0 and histogram[i] == 0:
            i+=1
            #print(i)
        start=i
        i=i+1
        while i < length:
            #print(i)
            while i<length and  histogram[i+1]!=0 and histogram[i] != 0:
                #print(i)
                i+=1
                
            index =i
            while i < length and histogram[i+1] == 0:
                #print(i)
                i=i+1
            #print(i,index)
            if (i-index > thresold):
                i+=1
                end = index
                sol.append((start,end))
                break
            i=i+1


    #print(sol)

    #print(histogram)

    # rolled = np.roll(histogram,1)
    # #print(rolled)


    # print(rolled.shape,histogram.shape)
    # cond1 = (rolled != histogram)
    # cond2 = rolled == 0
    # cond3 = histogram ==0
    # indices1 = np.where(cond1&cond2)[0]

    # print(indices1,"\n\n")

    # indices2 = np.where(cond1&cond3)[0]

    # print(indices2, "\n\n")

    # sol2=[]
    # for i in range(0,indices1.shape[0]):
    #     if ((i !=indices1.shape[0]-1) and (indices1[i+1]-indices2[i] > thresold+1)):
    #         sol2.append((indices1[i], indices2[i]))

    

    # print(sol,"\n\n\n",sol2)

    return sol



def getMaxTransitions(lineIm,baseLine):
    maxTransition=0
    maxTransitionIndex = 0

    width = lineIm.shape[1]

    for i in range(0, baseLine-1):
        currentTransition=0
        flag=0

        for j in range(0, width):
            if lineIm[i,j]==1 and flag==0:
                currentTransition+=1
            if lineIm[i,j]!=1 and flag==1:
                flag=0
            
                                  
        if currentTransition>=maxTransition:
            maxTransition=currentTransition
            maxTransitionIndex=i
            
    return maxTransitionIndex

def baseLineAndMaxLineDetection(lineIm,copp):

    his = np.sum(copp, axis=1)

    maxx=np.argmax(his)

    maxT = getMaxTransitions(copp, maxx)
    
    return maxx,maxT

def getMidCut(mfv,hisLine,start,midIndex,end,sr):

    #print ("start ",start,midIndex,end)
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

    
    


def cutPointIdentification(wordImage,wordImage2,MaxTransition):


    #wordImage2 = wordImage.copy()
    #wordImage2[wordImage == 255] = 1    # character
    #wordImage2[wordImage == 0]   = 0    # background

    startIndex = wordImage2.shape[1]
    for i in reversed(range(0, wordImage2.shape[1])):
        if wordImage2[MaxTransition,i]==1 :
            startIndex=i+1
            break

    hisLine = np.sum(wordImage2, axis=0)

    mfv=stats.mode(hisLine)[0][0]


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

            sr["mid"] = mid
            srList.append(sr)
        elif wordImage2[MaxTransition, i] != 1 and flag ==0:
            sr = {}
            sr["start"] = i
            flag = 1

    
    return srList


def noConnectedBaseLine(wordImage,baseLine,start,end): # two conditions because sad is above base line

    cond1=False
    imageSub=wordImage[baseLine,end+1:start]
    for i in range(0, len(imageSub)):
        if imageSub[i]==0:
            cond = True
            break
    cond2=False
    imageSub = wordImage[baseLine-1, end+1:start]
    for i in range(0, len(imageSub)):
        if imageSub[i] == 0:
            cond2 = True
            break

    return cond1 and cond2
    

def sheckSeenAndSheen(wordImage2,WordImageNoDots,validSeperations,seg,segn,segnn,baseLine,mfv):
    #print("in check seen and sad")
    if (len(seg) != 0)and isStroke(wordImage2,WordImageNoDots ,seg[0], seg[1], baseLine, mfv,errorMfv=4) and not dotBelowOrAbove(wordImage2, seg[0], seg[1],thres=3):
        #print("SEG Is stroke and No Dots")
        if (len(segn) != 0)and isStroke(wordImage2, WordImageNoDots, segn[0], segn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segn[0], segn[1]):
            #print("SEGN Is stroke and No Dots  ---> accpeted i=i+3") # seen
            #validSeperations.append(srl[i])
            return True
        if (len(segnn) != 0) and isStroke(wordImage2, WordImageNoDots, segn[0], segn[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, segn[0], segn[1]) and isStroke(wordImage2, WordImageNoDots, segnn[0], segnn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segnn[0], segnn[1]):
            #print( "SEGN Is stroke and Dots  and SEGNN is stroke and no dots ---> accpeted i=i+3") # sheen 
            #validSeperations.append(srl[i])
            return True


    return False


def isDal(wordImage2, start, end, baseLine,thres=4):

    #print ("in is Dal ")
    wordImage2=wordImage2[:,end:start]
    index=-1

    his=np.sum(wordImage2,axis=0)


    x_index=np.where(his>0)[0]
    if (len(x_index)==0):
        return False
    #print( x_index)

    x_index=x_index[0]
    

    column =wordImage2[:, x_index]
    #print(column)

    #print(np.where(column > 0))
    y_index = np.where(column > 0)[0][0]

    #print ("base line and index",baseLine , y_index)
    if baseLine - y_index < thres:
        return True

    return False


def checkYa2inEnd(wordImage2, start, end, baseLine):

    #print(start)
    #print("in check ya2 and end")

    img=wordImage2[:,end:start+1]
    img_white = img

    width = int(img_white.shape[1] * SCALE_PERCENT / 100)
    height = int(img_white.shape[0] * 100 / 100)
    dim = (width, height)

    binary3 = img_white.copy()

    binary3[binary3 == 0] = 255
    binary3[binary3 == 1] = 0

    resized = cv2.resize(binary3, dim, interpolation=cv2.INTER_AREA)
    resizedcopy = resized.copy()

    #showScaled(resizedcopy, "resized", 400)

    resized[resizedcopy > 128] = 0
    resized[resizedcopy <= 128] = 1

  

    labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(
        resized,)
    maxLabel = []
    maxx = -1
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        #print(size)
        if size > 0 :
            if (y>baseLine+3 and x<((12* SCALE_PERCENT) / 100)):
                maxLabel.append(label)
                maxx = size
    for label in range(1, labelnum):
        x, y, w, h, size = contours[label]
        if label not in maxLabel:
            #resizedcopy[y:y+h, x:x+w] = 255
            pass


    
    return len(maxLabel)>0


    



def seperationFiltrations(wordImage,wordImage2,srl, baseLine,maxTransition):
    #wordImage2 = wordImage.copy()
    #wordImage2[wordImage == 255] = 1    # character
    #wordImage2[wordImage == 0]   = 0    # background

    WordImageNoDots=removeDotsAndHamza2(wordImage2)


    hisLine = np.sum(wordImage2, axis=0)


    mfv = stats.mode(hisLine)[0][0]


    srl = removeCutInHoles(wordImage2, wordImage,
                           WordImageNoDots, hisLine, srl, mfv, baseLine)


    validSeperations=[]


    i=0
    srlLength = len(srl)
    while i < srlLength:
        start = srl[i]["start"]
        end = srl[i]["end"]
        #print("*******************************cut Number = ",i,"  start", "end = " ,start,end, "mfv = ", mfv)

        serp=[]
        seg=[]
        segn=[]
        segnn=[]
        if  i ==0 or i ==srlLength-1:
            serp.append(srl[i]["start"])
            serp.append(srl[i]["end"])
        else:
            serp.append(srl[i-1]["mid"])
            serp.append(srl[i+1]["mid"])
        
        if i!=srlLength-1:
            seg.append(srl[i]["mid"])
            seg.append(srl[i+1]["mid"])

        if i < srlLength-2 :
            segn.append(srl[i+1]["mid"])
            segn.append(srl[i+2]["mid"])
        if i < srlLength-3:
            segnn.append(srl[i+2]["mid"])
            segnn.append(srl[i+3]["mid"])



        if hisLine[srl[i]["mid"]]==0:
            #print("cond 1")
            validSeperations.append(srl[i])
            if sheckSeenAndSheen(wordImage2, WordImageNoDots, validSeperations, seg, segn, segnn, baseLine, mfv):
                i=i+3
            else :
                i+=1
        elif  noConnectedBaseLine(wordImage2, baseLine, start, end):
            #print("cond 3")
            if (underBaseLine(wordImage2[:, end: start] ,baseLine)):
                i+=1
                continue
            elif hisLine[srl[i]["mid"]] < mfv :
                validSeperations.append(srl[i])
                i += 1
            else :
                i+=1
                continue
        elif (srl[i] == srl[-1] and srl[i]["mid"]<9 and  checkSegmentLength(wordImage2, srl[i]["mid"], 0, baseLine)):
            i+=1
            continue
        elif (i <srlLength-1 and hisLine[srl[i+1]["mid"]]==0 and isDal(wordImage2, seg[0], seg[1], baseLine,thres=5)): 
            validSeperations.append(srl[i])
            i+=1
            continue
        elif (i == 0) and (srlLength > 2) and sheckSeenAndSheen(wordImage2, WordImageNoDots, validSeperations, [wordImage.shape[1]-1, srl[i]["mid"]], [srl[i]["mid"], srl[i+1]["mid"]], [srl[i+1]["mid"], srl[i+2]["mid"]], baseLine, mfv):
            i=i+2
            continue
        # or ((len(seg)!=0) and (len(segn)!=0) and not hasHole(wordImage2, seg[0], seg[1]) and not hasHole(wordImage2, segn[0], segn[1])and  hasHole(wordImage2, seg[0], segn[1])):
        elif ((len(seg) != 0) and not isStroke(wordImage2, WordImageNoDots, seg[0], seg[1], baseLine, mfv)):
            #print("SEG not stroke ")
            if  (i <srlLength-1) and noConnectedBaseLine(wordImage2, baseLine, srl[i+1]["start"], srl[i+1]["end"]) and hisLine[srl[i+1]["mid"]] <= mfv:
                #print("SEG no stroke and noConnectedBaseLine")
                i+=1
                continue
            else:
                #print("SEG no stroke and ConnectedBaseLine ---> accepted i=i+1")
                validSeperations.append(srl[i])
                i += 1
                continue
        elif (len(seg) != 0)and isStroke(wordImage2,WordImageNoDots, seg[0], seg[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, seg[0], seg[1]):
                #print("SEG Is stroke and Dots ----> accepted i=i+1")
                validSeperations.append(srl[i])
                i=i+1
        elif (len(seg) != 0)and isStroke(wordImage2,WordImageNoDots ,seg[0], seg[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, seg[0], seg[1],thres=3):
            #print("SEG Is stroke and No Dots")
            if (len(segn) != 0)and isStroke(wordImage2,WordImageNoDots, segn[0], segn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segn[0], segn[1]):
                #print("SEGN Is stroke and No Dots  ---> accpeted i=i+3")
                validSeperations.append(srl[i])
                i=i+3
                continue
            if (len(segnn) != 0) and isStroke(wordImage2, WordImageNoDots, segn[0], segn[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, segn[0], segn[1]) and isStroke(wordImage2, WordImageNoDots, segnn[0], segnn[1], baseLine, mfv) and not dotBelowOrAbove(wordImage2, segnn[0], segnn[1]):
                #print( "SEGNN Is stroke and Dots  and SEGNN is stroke and no dots ---> accpeted i=i+3")
                validSeperations.append(srl[i])
                i = i+3
                continue
            if (len(segn) != 0)and ( not isStroke(wordImage2, WordImageNoDots,segn[0], segn[1], baseLine, mfv) or (isStroke(wordImage2,WordImageNoDots, segn[0], segn[1], baseLine, mfv) and dotBelowOrAbove(wordImage2, segn[0], segn[1]))):
                #print("SEGN Is not  stroke  or segn is stroke with Dots ")
                i=i+1
                continue
            i=i+1
        else :
            #print("cond else")
            validSeperations.append(srl[i])
            i += 1

    #copyyy = wordImage.copy()
    #printCut(copyyy, validSeperations)
    #showScaled(copyyy,"before reurn some cuts",400)

    return returnSomeCuts(wordImage2, validSeperations,baseLine)



def returnSomeCuts(wordImage2, validSeperations,baseLine):

    if len(validSeperations)>0:
        if len(validSeperations) > 1 and validSeperations[-1]["mid"] > 12 and checkYa2inEnd(wordImage2, validSeperations[-1]["mid"], 0, baseLine):
            sr = {}
            sr["mid"] = 10
            validSeperations.append(sr)


        # if there is as space between in segment but a cut in it 

        def appendZeroCuts(sol,validSeperations,i,part):
            zero_indices = np.where(part == 0)[0].tolist()
            sol.append(validSeperations[i])
            for j in reversed(range(len(zero_indices))):
                sr = {}

                if zero_indices[j] < 2 or zero_indices[j] > len(part) -2 :
                    continue
                if (i==-1):
                    sr["mid"] = zero_indices[j]+0
                else:
                    sr["mid"] = zero_indices[j]+validSeperations[i+1]["mid"]

            
                sol.append(sr)


        hisline = np.sum(wordImage2, axis=0)
        sol=[]
        i=0
        while i < len(validSeperations):
            #print("valid i",i, hisline[validSeperations[i]["mid"]])
            if ( i <len(validSeperations)-1)and \
                hisline[validSeperations[i]["mid"]] != 0 and\
                hisline[validSeperations[i+1]["mid"]]!=0:
                part = hisline[validSeperations[i+1]["mid"]:validSeperations[i]["mid"]]
                appendZeroCuts(sol,validSeperations,i,part)

            elif( i <len(validSeperations)-1):
                sol.append(validSeperations[i])
            
            i=i+1

        if len(validSeperations)>0:
            part = hisline[0:validSeperations[-1]["mid"]]
            appendZeroCuts(sol,validSeperations,-1,part)

        #print(sol)

                
                

        validSeperations=sol
    else:
        if wordImage2.shape[1] > 12 and checkYa2inEnd(wordImage2, wordImage2.shape[1]-1, 0, baseLine):
            sr={}
            sr["mid"]=10
            validSeperations.append(sr)
        else:
            index = wordImage2.shape[1]-1-10
            while 0 <index:
                sr={}
                sr["mid"] = index
                #print("xxxxxxxxxxxxxxxxxxxx i =0 ")
                validSeperations.append(sr)
                index=index-10

    

    valid2=[]
    for i in range(0, len(validSeperations)):
        #print("mid i=", i, " value = ",
        #      validSeperations[i]["mid"], "  ", wordImage2.shape[1]-1, (wordImage2.shape[1]-1 - validSeperations[i]["mid"] >20))
        if (i == 0 and (wordImage2.shape[1]-1 - validSeperations[i]["mid"]>20)):
            index=wordImage2.shape[1]-1-10
            while validSeperations[i]["mid"] <index:
                sr={}
                sr["mid"] = index
                #print("xxxxxxxxxxxxxxxxxxxx i =0 ")
                valid2.append(sr)
                index=index-10
        elif (i==0):
            valid2.append(validSeperations[i])
        elif (i>0) and (i<len(validSeperations)-1) and validSeperations[i]["mid"] - validSeperations[i+1]["mid"] > 20:
            index = validSeperations[i]["mid"]-10
            valid2.append(validSeperations[i])
            while validSeperations[i+1]["mid"] < index:
                sr = {}
                sr["mid"] = index
                #print("xxxxxxxxxxxxxxxxxxxx")

                valid2.append(sr)
                index = index-10
        elif (i>0) and (i<len(validSeperations)-1):
            valid2.append(validSeperations[i])
        elif (validSeperations[i] == validSeperations[-1])and (validSeperations[i]["mid"] <= 20):
            valid2.append(validSeperations[i])
        elif (validSeperations[i] == validSeperations[-1]) and (validSeperations[i]["mid"] >20):
            valid2.append(validSeperations[i])
            index = validSeperations[i]["mid"]-10
            while 0 < index:
                sr = {}
                sr["mid"] = index
                #print("xxxxxxxxxxxxxxxxxxxx i =end")

                valid2.append(sr)
                index = index-10



    #print(valid2)
    return valid2
    

def removeCutInHoles(wordImage2, wordImage, WordImageNoDots, hisLine, srl, mfv, baseLine):
    validSeperations = []
    i=1
    copyt=[]
    srlLength=len(srl)
    if srlLength >= 2 and not (hasHole(wordImage2,WordImageNoDots ,wordImage2.shape[1]-1, srl[1]["start"]) and CutsInHole(wordImage2,  srl[0]['start'],  srl[0]['mid'],  srl[0]['end'], baseLine)):
        copyt .append (srl[0])
    
    while i<srlLength-1:
        serp = []
        serp.append(srl[i-1]["mid"])
        serp.append(srl[i+1]["mid"])
        if hisLine[srl[i]["mid"]] == 0 or noConnectedBaseLine(wordImage2, baseLine, srl[i]["start"], srl[i]["end"]):
            #print("space")
            copyt.append(srl[i])
        elif hasHole(wordImage2, WordImageNoDots, serp[0], serp[1]) and not (hasHole(wordImage2, WordImageNoDots, serp[0], srl[i]["mid"]) and hasHole(wordImage2, WordImageNoDots, srl[i]["mid"], serp[1])):
            #print("can be hole")
            if CutsInHole(wordImage2,  srl[i]['start'],  srl[i]['mid'],  srl[i]['end'],baseLine):
                #print("is hole")
                #print ('i+',i)
                pass
            else :
                #print("not Hole")
                copyt.append(srl[i])
        else :
            #print("not any thing")
            copyt.append(srl[i])
            
        i=i+1
    #print("before cut in hole")
    if srlLength >= 2 and not (hasHole(wordImage2, WordImageNoDots, srl[-2]["mid"], 0) and CutsInHole(wordImage2,  srl[-1]['start'],  srl[-1]['mid'],  srl[-1]['end'], baseLine)):
        copyt .append(srl[-1])

    srl = copyt
    #print ("length new ",len(copyt),len(srl))

    #ww=wordImage.copy()
    #printCut(ww, srl)
    #showScaled(ww,"shehab",400)

    return srl




def dotBelowOrAbove(img, start, end,thres=1):
    if thres >= 1 :
        img=removeDotsAndHamza2(img[:,end:start+1],thres)
    else:
        img=img[:,end:start+1]

    #ww = img.copy()
    #ww[ww == 0] = 255
    #ww[ww == 1] = 0

    
    #showScaled(ww, "after remove in dots", 400)

    _, contours, _ = cv2.findContours(img.copy(),
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    return len(contours) > 1


def isStroke(wordImage2,noDots, start, end, baseLine, mfv, alfLength=13, error=5, errorMfv=2):
    img = wordImage2[:,end:start].copy()

    #print("*******is Stroke function results****")
    
    #noDots=removeDotsAndHamza2(wordImage2)
    noDots = noDots[:, end:start]
    labelnum, _, _, _ = cv2.connectedComponentsWithStats(noDots)
    #print("label number",labelnum)
    if labelnum >2:
        return False
    #print("is single connected conponens")

    if  underBaseLine(wordImage2[:, end:start],baseLine):
        return False
    #print("Not under base Line")


    hisLine = np.sum(wordImage2[:, end:start], axis=1)  # اhoriontal histogram
    hisLine = hisLine[hisLine>0]


    mfvHorizontal = stats.mode(hisLine)[0]

    if len(mfvHorizontal)==0:
        return False

    mfvHorizontal=mfvHorizontal[0]



    if (abs(int(mfvHorizontal)- int(mfv)) >errorMfv ):
        return False
 

    if hasHole(wordImage2, noDots, start, end):
        return False

    #print("No hole")

    biggest =biggestConnectedComponent(wordImage2[:, end:start])
    h=calculateHeight(biggest)
    #print("h equal", h)

    if (h>alfLength-error ):
        return False

    #print("Stroke lenth is good")


    
    


    return True


def checkSegmentLength(wordImage2, start, end, baseLine,thres=6):

    #print ("in segment lenth function")

    wordImageNoDots=removeDotsAndHamza2(wordImage2)

    hisLine = np.sum(wordImageNoDots[:, 0:start], axis=1)  # vertical histogram

    indx = np.where(hisLine > 0)[0]

    if len(indx) <=0 :
        return False
    mini = indx[0]
    maxi = indx[-1]
    
    #print(abs((maxi-mini))," length")
    if abs((maxi-mini)) <thres :
        return True
    elif abs((maxi-mini)) < thres+2:
        column = wordImage2[:, start+1]
        y_indexUp = np.where(column > 0)[0][0]
        y_indexDown= np.where(column > 0)[0][-1]

        #print(y_indexUp, baseLine, "dfdsfsd")

        if y_indexUp < baseLine-2:
            return True 
        if y_indexDown > baseLine+2:
            return True


    
    return False



def underBaseLine(wordImage2 ,baseLine):

    hisLine = np.sum(wordImage2, axis=1)  # horizontal histogram


    sumOveBbaseLine = np.sum(hisLine[0:baseLine])


    sumUnderBaseLine = np.sum(hisLine[baseLine+1:])


    if sumUnderBaseLine > sumOveBbaseLine :
        return True
    
    return False


def CutsInHole(wordImage,start,midle,end,baseLine,offset =2):
    wordImage=removeDotsAndHamza2(wordImage)

    count=0
    countUnderBase=0
    for i in range(1,wordImage.shape[0]):
        if wordImage[i, midle] == 1 and wordImage[i-1, midle]==0:
            if i >= offset+baseLine :
                countUnderBase+=1
            count +=1
        if wordImage[i, midle] == 0 and wordImage[i-1, midle] == 1:
            if i >= offset+baseLine:
                countUnderBase+=1
            count +=1

    #print (count)

    if count ==0 :
        return False
    if count == countUnderBase :
        return False
    return  ( wordImage[wordImage.shape[0]-1, midle] ==1 and  count !=1 ) or( wordImage[wordImage.shape[0]-1, midle] ==0 and  count !=2) 
    


def printCut(img,srl):
    for sr in srl:
        binary = cv2.line(img, (sr["mid"], 0),
                      (sr["mid"], img.shape[1]-1), 128, 1)

    
    

def removeDotsAndHamza2(img,thres=15):  # must be less than 20 (to handle r case)
    img_white = img.copy()


    width = int(img_white.shape[1] * SCALE_PERCENT / 100)
    height = int(img_white.shape[0] * 100 / 100)
    dim = (width, height)

    binary3 = img_white.copy()

    binary3[binary3 == 0] = 255
    binary3[binary3 == 1] = 0

    binary33 = np.zeros((binary3.shape[0], binary3.shape[1]+1))
    binary33[:, :-1] = binary3
    binary33[:, -1] = 255
    binary3=binary33
    resized = cv2.resize(binary3, dim, interpolation=cv2.INTER_AREA)
    resizedcopy = resized.copy()

    #showScaled(resizedcopy, "resized", 400)


    resized[resizedcopy > 128] = 0
    resized[resizedcopy <= 128] = 1

    ## words segminations

    sol2 = segmentation(np.sum(resized, axis=0), 0)

    #print("len ofsub",len(sol2))


    for hor in sol2:
        begin2, end2 = hor
        begin2 = round(begin2*(100.0/SCALE_PERCENT))
        end2 = round(end2*(100.0/SCALE_PERCENT))

        sub = img_white[:, begin2:end2]

        labelnum, labelimg, contours, GoCs = cv2.connectedComponentsWithStats(sub,)
        
        maxLabel =[]
        maxx=-1
        for label in range(1, labelnum):
            x, y, w, h, size = contours[label]
            #print(size)
            if size > thres:
                maxLabel.append(label)
                maxx=size
        for label in range(1, labelnum):
            x, y, w, h, size = contours[label]
            if label not in  maxLabel:
                sub[y:y+h, x:x+w] = 0


    #ww=img_white.copy()
    #ww[ww==0]=255
    #ww[ww==1]=0

    #showScaled(ww,"image without dots",400)
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
    

def hasHole(wordImage, WordImageNoDots, start, end):

    if start ==-1 and end ==-1 :
        return False
    wordImage = WordImageNoDots

    _, contours, _ = cv2.findContours(wordImage[:, end:start+1].copy(),
                     cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    return  len(contours) > 1


def charSegmentation(binary):

    AllImageRealOnes = binary.copy()
    AllImageRealOnes[binary == 255] = 0 #background
    AllImageRealOnes[binary == 0] = 1  # character 

    sol = segmentation(np.sum(AllImageRealOnes, axis=1), 1)
    wordList = []
    for line in sol:
        begin, end = line

        width = int(AllImageRealOnes[begin:end][:].shape[1] * SCALE_PERCENT / 100)
        height = int(AllImageRealOnes[begin:end][:].shape[0] * 100 / 100)
        dim = (width, height)

        resized = cv2.resize(binary[begin:end][:],
                             dim, interpolation=cv2.INTER_AREA)
        resizedcopy = resized.copy()

        resized[resizedcopy > 128] = 0
        resized[resizedcopy < 128] = 1

        ## words segminations
        sol2 = segmentation(np.sum(resized, axis=0), 8)
        lineList = []
        for hor in sol2:
            begin2, end2 = hor

            begin2 = round(begin2*(100.0/SCALE_PERCENT))
            end2 = round(end2*(100.0/SCALE_PERCENT))

            if np.all(binary[begin:end, begin2:end2] == 0):
                continue


            baseLine, maxLine = baseLineAndMaxLineDetection(
                binary[begin:end, begin2:end2], AllImageRealOnes[begin:end, begin2:end2])
            
            # white font image now in the word postion
            binary[begin:end, begin2:end2] = 255 - \
                binary[begin:end, begin2:end2]

            wordWhiteText = binary[begin:end, begin2:end2]
            wordBinary = AllImageRealOnes[begin:end, begin2:end2]
            srl = cutPointIdentification(
                wordWhiteText, wordBinary, maxLine)


            imgWithAllCuts = binary[begin:end, begin2:end2].copy()
            #showScaled(imgWithAllCuts, "Puer image", 400)

            #printCut(imgWithAllCuts, srl)

            srl = seperationFiltrations(
                wordWhiteText, wordBinary, srl, baseLine, maxLine)

            #printCut(binary[begin:end, begin2:end2], srl)

            #showScaled(imgWithAllCuts,"image with All cuts", 400)
            #showScaled(binary[begin:end, begin2:end2] ,"image with correct filtrarions", 400)

            word = {'rows': (begin, end), 'columns': (
                begin2, end2), "srl": srl}
            lineList.insert(0, word)

        wordList = wordList+lineList

    return wordList

def showWordCuts(img,wordList,color=128):
    binary =img
    for word in wordList:
        (begin, end) = word["rows"]
        (begin2, end2) = word["columns"]

        binary = cv2.line(binary, (begin2, begin),
                          (begin2, end), color, 1)
        binary = cv2.line(binary, (end2, begin),
                          (end2, end), color, 1)
        binary = cv2.line(binary, (begin2, begin),
                                  (end2, begin), color, 1)
        binary = cv2.line(binary, (begin2, end),
                          (end2, end), color, 1)

    cv2.imwrite("word_segmented.png", binary)





def main():

    for i in range (0,1):

        file=random.choice(os.listdir("scanned\\"))
        img = cv2.imread("capr2.png")#"scanned\\"+file)

        thre = binarize(img)
        rotated = textSkewCorrection(thre)
        wordList=charSegmentation(rotated)
        print(len(wordList))

                
                

main()







