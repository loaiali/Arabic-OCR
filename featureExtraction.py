import cv2
import numpy as np
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Preprocessing import preprocessImageFromPath as preprocessImage, binarize
import os
import glob 
from config import featuresDir
'''
There are 3 types of features
1-Structural features:will be number of dots,number of end points,number of loops,
2-Statistical features: will be number of connected components,zoning features
3-Global Transoformation :DCT,HOG
'''

def getNumDots(characterImage):
    numDots=0
    contours=cv2.findContours(characterImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    maxDotArea=20
    minDotArea=3
    for contour in contours:
        if minDotArea<cv2.contourArea(contour)<maxDotArea:
            numDots+=1
    return numDots

def getNumberLoops(characterImage):
    _,contours,_=cv2.findContours(characterImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def getNumberEndPoints(characterImage):
    if np.max(characterImage)>1: 
        characterImage=np.where(characterImage>=255,1,0)
    #print(characterImage)
    skel=skeletonize(characterImage)
    skel = skel.copy()
    #cv2.imshow("skel",skel*255.0)
    #cv2.waitKey(0)
    skel[skel!=0] = 1
    skel = np.uint8(skel)
    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 1
    _,count=np.unique(out,return_counts=True)
    # print(f"count.shape: {count.shape}")
    if len(count)==1:
        return 0
    return count[1]  

def getConnectedComponents(characterImage):
    connectivity=8
    output = cv2.connectedComponentsWithStats(characterImage, connectivity, cv2.CV_32S)
    return output[0] #this is the number of labels 

def get4ZonesFeatures(characterImage):
    firstQuarterDiagonal=characterImage[0:len(characterImage[0])//2][0:len(characterImage[1])//2].diagonal()
    seconedQuarterDiagonal=characterImage[len(characterImage[0])//2:len(characterImage[0])][0:len(characterImage[1])//2].diagonal()
    thirdQuarterDiagonal=characterImage[0:len(characterImage[0])//2][len(characterImage[1])//2:len(characterImage[1])].diagonal()
    fourthQuarterDiagonal=characterImage[len(characterImage[0])//2:len(characterImage[0])][len(characterImage[1])//2:len(characterImage[1])].diagonal()
    diagonalSum=[np.sum(firstQuarterDiagonal)/255,np.sum(seconedQuarterDiagonal)/255,np.sum(thirdQuarterDiagonal)/255,np.sum(fourthQuarterDiagonal)/255]
    return diagonalSum
def get16VerticalZonesFeatures(characterImage):
    blocks=[]
    offsetX=len(characterImage)//16
    currentX=0
    #if we want to make 16 partition we need  
    for _ in range (0,15):
        blocks.append(characterImage[currentX:currentX+offsetX][:])
        currentX+=offsetX    
    blocksSum=[]
    for block in blocks:
        blocksSum.append(np.sum(block)/255)
    return blocksSum

def get16HorizontalZonesFeatures(characterImage):
    blocks=[]
    offsetX=len(characterImage)//16
    currentX=0
    #if we want to make 16 partition we need  
    for _ in range (0,15):
        blocks.append(characterImage[:][currentX:currentX+offsetX])
        currentX+=offsetX    
    blocksSum=[]
    for block in blocks:
        blocksSum.append(np.sum(block)/255)
    return blocksSum

def getDctCoeff(characterImage):
    imageFloat=np.float32(characterImage)/255.0  
    imageFloat=cv2.resize(imageFloat,(200,200))
    dst = cv2.dct(imageFloat)           # the dct
    dctCoeff = np.uint8(dst)*255.0    # convert back
    dctVector=[[] for i in range(len(dctCoeff[1])+len(dctCoeff[0])-1)]
    for i in range(0,len(dctCoeff[0])): 
        for j in range(0,len(dctCoeff[1])): 
            sum=i+j 
            if(sum%2 ==0): 
                #add at beginning 
                dctCoeff[i][j]
                dctVector[sum].insert(0,dctCoeff[i][j]) 
            else: 
                #add at end of the list 
                dctVector[sum].append(dctCoeff[i][j]) 
    dctVector = [item for sublist in dctVector for item in sublist]
    return dctVector[0:200] 

def getHogFeatures(characterImage):
    characterImage=cv2.resize(characterImage,(64,64))
    #hogComputer = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winSize = (64,64)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (16,16)
    nbins = 9
    hogComputer=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    
    x=hogComputer.compute(cv2.Canny(characterImage,0,255))
    return x    

def extractFeatures(characterImage):
    featureVector=[getNumDots(characterImage),getNumberLoops(characterImage),getNumberEndPoints(characterImage),getConnectedComponents(characterImage)]
    featureVector.extend(get4ZonesFeatures(characterImage))
    featureVector.extend(get16HorizontalZonesFeatures(characterImage))
    featureVector.extend(get16VerticalZonesFeatures(characterImage))
    featureVector.extend(getDctCoeff(characterImage))
    featureVector.extend(getHogFeatures(characterImage).flatten())
    
    #normalize feature vector
    maxFeatureValue=max(featureVector)
    minFeatureValue=min(featureVector)
    for i in range (0,len(featureVector)):
        featureVector[i]=(featureVector[i]-minFeatureValue)/(maxFeatureValue-minFeatureValue)
    return featureVector 
def writeFeatureVector(dir,features):
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    outputFile=open(dir,'w')
    # print(f"Feature vector shape: {len(features)}")
    for feature in features:
        outputFile.write(str(feature))
        outputFile.write("\n")
    outputFile.close()


def imageToFeatureVector(imagePath):
    img = cv2.imread(f)
    img = binarize(img)
    return img.flatten()
    # characterImage=preprocessImage(imagePath)
    # return extractFeatures(characterImage)

def imgToFeatureVector(image):
    from Preprocessing import preprocessImage as pi
    characterImage=pi(image)
    return extractFeatures(characterImage)

if __name__=="__main__":
    folders = glob.glob('dataset\\*')
    for folder in folders:
        print(f"currently in dire: {folder}")
        for test_case in glob.glob(folder+'/*'):    
            print(f"folder: {test_case}")
            for f in glob.glob(test_case+'/*.png'):
                outputFolder= featuresDir + f[7:-4]+".txt"
                # characterImage=preprocessImage(f)
                img = cv2.imread(f)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                featureVector=extractFeatures(img)
                
                writeFeatureVector(outputFolder,featureVector)
