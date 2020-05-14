import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__),"..", "preprocessing")) 
sys.path.append(os.path.join(os.path.dirname(__file__),"..", "segmentation")) 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from Preprocessing import textSkewCorrection, binarize
from scipy import ndimage
from segmentation import  showScaled, showWordCuts, charSegmentation
from wordseg import wordSegmentation
import pyarabic.araby as araby
import pyarabic.number as number
import nltk
from progressbar import *
from PIL import ImageFont, ImageDraw, Image
import textwrap
import pickle

englishName = {}
with open('char_names.pickle', 'rb') as handle:
    englishName = pickle.load(handle)


nonConnChOneSide = ["و", "ر", "ز", "ذ", "د",
                    "ا", "أ", "إ", "ة", "ء", "؟", ".", ",", '،', "لأ", "لإ","لا","(",")"]
nonConnChTwoSide = ["ء", "؟", ".", ",", '،',"(",")"]


def searchTextOnly():
    f=open("textOnly.txt","w")
    for file in os.listdir("scanned//"):
        file = file.replace(".png", "")
        ftext = open("text\\"+file+".txt", "r", encoding="utf-8")
        text = ftext.read()

        special = ["؟", "0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9", "..", "(", ")", ",", '،']

        if not any(True for ch in special if ch in text):
            print(file)
            f.write(file+".txt\n")

        
    f.close()

    pass


def write_tokenz(list):
    f = open("last token.txt", "w", encoding="utf-8")
    for word in list:
        f.write(word+"\n")

    f.close()


def check_on_length(wrong_file,right_file,ending=2000):
    fWrong = open(wrong_file,"w")
    fRight = open(right_file, "w")

    cadidate = open("textOnly.txt")
    cadidate=cadidate.readlines()
    total=0
    wrong = 0
    widgets = ['Test: ', Percentage(), ' ', Bar(marker='0', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]  # see docs for other options

    pbar = ProgressBar(widgets=widgets, maxval=ending)
    pbar.start()
    iii=0
    for file in cadidate:
        file=file.rstrip()
        print(file)
        pbar.update(iii)
        iii += 1
        file=file.replace(".txt","")
        img = cv2.imread("scanned\\"+file+".png")
        ftext = open("text\\"+file+".txt", "r", encoding="utf-8")
        ## image segmentation


        text = ftext.read()
        wordsArray = nltk.word_tokenize(text)


        thre = binarize(img)
        rotated = textSkewCorrection(thre)
        wordList = wordSegmentation(rotated)


        print(file,len (wordsArray), len(wordList))


        
        total+=1
        
        if len(wordList) != len (wordsArray):
            wrong +=1
            stt="{0:<20}  {1:<20}  {2:<20}".format(
                file, str(len(wordList)), str(len(wordsArray)))
            fWrong.write(stt+"\n")
            print("************wrong Image************")
            write_tokenz(wordsArray)
            #showWordCuts(img,wordList)
            break
        else:
            fRight.write(file+".txt"+"\n")
            print("************corrent Image************")

        
        ftext.close()
        if total == ending:
            break
    
    fWrong.close()
    fRight.close()
    pbar.finish()




def getCharacterLength(chDict, nonConnChOne, nonConnChTwo, word, index):
    if index == 0 and (len(word) != 1 and word[index+1] not in nonConnChTwo):
        return chDict[word[0]][0]
    elif index == 0 and (len(word) != 1 and word[index+1] in nonConnChTwo):
        return chDict[word[0]][0]
    elif index==0 and len(word)==1:
        return chDict[word[0]][3]
    elif index == len(word)-1 and word[index-1] not in nonConnChOne:
        return chDict[word[index]][2]
    elif index == len(word)-1 and word[index-1] in nonConnChOne:
        return chDict[word[index]][3]
    elif  word[index-1] not in  nonConnChOne and word[index+1] not in nonConnChTwo:
        return chDict[word[index]][1]
    elif  word[index-1] in nonConnChOne and word[index+1] not in nonConnChTwo:
        return chDict[word[index]][0]
    elif word[index-1] not in nonConnChOne and word[index+1] in nonConnChTwo:
        return chDict[word[index]][2]
    elif word[index-1] in nonConnChOne and word[index+1]  in nonConnChTwo:
        return chDict[word[index]][3]


def checkLamAlf(word,index):
    if index == 0:
        return False
    return ( (word[index] == 'ل' )and (index != len(word)-1)and(word[index+1] == 'ا' or word[index+1] == 'أ' or word[index+1] == 'إ' or word[index+1] == 'آ'))



def createDataSet(img,wordList,wordsArray,nonConnChOneSide,nonConnChTwoSide,englishName,file):

    c=0
    for Wordimage,word in zip(wordList, wordsArray):
        rows=Wordimage["rows"]
        columns = Wordimage["columns"]
        srl=Wordimage["srl"]
        image = img[rows[0]:rows[1], columns[0]:columns[1]]
        wordLength=0
        i=0
        characters=[]

        while i < len(word):
            if (checkLamAlf(word, i)):
                characters.append(word[i:i+2])
                i=i+2
            else:
                characters.append(word[i])
                i=i+1 
            wordLength += 1
        
        if wordLength == len(srl)+1:
            i=0
            c=0
            print(characters,len(srl)+1)
            for indx,char in enumerate(characters):
                charImage=None
                if indx==0:
                    charImage = image[:, srl[indx]["mid"]:]
                elif indx <len(characters)-1:
                    charImage = image[:, srl[indx]["mid"]:srl[indx-1]["mid"]]
                else:
                    charImage = image[:, 0:srl[indx-1]["mid"]]      
                        
                print('dataset\\'+ englishName[char]+"\\"+file)
                if not os.path.exists('dataset\\'+englishName[char]+"\\"+file):
                    os.makedirs('dataset\\'+englishName[char]+"\\"+file)

                fileName = 'dataset\\' +englishName[char]+"\\"+file+"\\" + str(c)+".png"
                cv2.imwrite(fileName, charImage)
                c+=1


check_on_length("wrong.txt","right.txt",ending=5000)
f = open("right.txt", "r")
fw = open("output.txt", "w", encoding="utf-8")
files = f.read().split()

for file in files:

       img = cv2.imread("scanned\\"+file.replace(".txt","")+".png")
       ftext = open("text\\"+file, "r", encoding="utf-8")
       ## image segmentation

        # text preprocessing
       text = ftext.read().replace(":", "").replace(".", "").replace("-","")

        # tokenizations
       wordsArray = nltk.word_tokenize(text)
       for word in wordsArray:
           fw.writelines([word, "  ", str(word.encode("utf8-")), "\n"])

       thre = binarize(img)
       rotated = textSkewCorrection(thre)

       wordList = charSegmentation(rotated)

       createDataSet(rotated,wordList, wordsArray,
                     nonConnChOneSide, nonConnChTwoSide, englishName, file)
        

f.close()
fw.close()




            













                



