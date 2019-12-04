import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from Preprocessing import textSkewCorrection, binarize
from scipy import ndimage
from segmentation import wordSegmentation
import pyarabic.araby as araby
import pyarabic.number as number
import nltk

characterDict={}

characterDict["أ"] = (4, 4, 4, 4)
characterDict["آ"] = (4, 4, 4, 4)
characterDict["ا"] = (4, 4, 4,4)
characterDict["إ"] = (4, 4, 4,4)
characterDict["ب"] = (6, 6, 14, 14)
characterDict["ت"] = (6, 6, 14, 14)
characterDict["ث"] = (6, 6, 14, 14)
characterDict["ج"] = (12, 10, 11, 11)
characterDict["ح"] = (12, 10, 11, 11)
characterDict["خ"] = (12, 10, 11, 11)
characterDict["د"] = (9, 9, 9, 9)
characterDict["ذ"] = (9, 9, 9, 9)
characterDict["ر"] = (6, 6, 6, 6)
characterDict["ز"] = (6, 6, 6, 6)
characterDict["س"] = (14, 13, 18, 18)
characterDict["ش"] = (14, 13, 18, 18)
characterDict["ص"] = (16, 16, 20, 20)
characterDict["ض"] = (16, 16, 20, 20)
characterDict["ط"] = (12, 12, 12, 12)
characterDict["ظ"] = (12, 12, 12, 12)
characterDict["ع"] = (10, 9, 10, 10)
characterDict["غ"] = (10, 9, 10, 10)
characterDict["ق"] = (9, 9, 15, 15)
characterDict["ف"] = (9, 9, 15, 15)
characterDict["ك"] = (10, 10, 14, 14)
characterDict["ل"] = (6, 6, 11, 11)
characterDict["م"] = (9, 10, 9, 9)
characterDict["ن"] = (5, 6, 11, 11)
characterDict["ه"] = (11, 10, 8, 11)
characterDict["و"] = (8, 8, 8, 8)
characterDict["ي"] = (7, 8, 12, 8)
characterDict["ة"] = (11, 10, 8, 6)
characterDict["ئ"] = (11, 7, 8, 6)
characterDict["ؤ"] = (8, 8, 8, 8)
characterDict["ى"] = (7, 8, 12, 8)
characterDict["ء"] = (7, 7, 7, 7)
characterDict["لا"] = (8, 8, 8,8)
characterDict["لأ"] = (8, 8, 8,8)
characterDict["لإ"] = (8, 8, 8, 8)
characterDict["لآ"] = (8, 8, 8, 8)
characterDict["؟"] = (7, 7, 7, 7)
characterDict["."] = (4, 4, 4, 4)
characterDict[","] = (4, 4, 4, 4)
characterDict['،'] = (4, 4, 4, 4)
characterDict['1'] = (9, 9, 9, 9)
characterDict['2'] = (9, 9, 9, 9)
characterDict['3'] = (9, 9, 9, 9)
characterDict['4'] = (9, 9, 9, 9)
characterDict['5'] = (9, 9, 9, 9)
characterDict['6'] = (9, 9, 9, 9)
characterDict['7'] = (9, 9, 9, 9)
characterDict['8'] = (9, 9, 9, 9)
characterDict['9'] = (9, 9, 9, 9)
characterDict['0'] = (9, 9, 9, 9)

characterDict['('] = (6, 6, 6, 6)
characterDict[')'] = (6, 6, 6, 6)




englishName = {}


englishName["أ"] = "alfHamzaFo2"
englishName["آ"] = "alfMad"
englishName["ا"] = "alf"
englishName["إ"] = "alfHamzaTa7t"
englishName["ب"] = "ba2"
englishName["ت"] = "ta2"
englishName["ث"] = "tha2"
englishName["ج"] = "geem"
englishName["ح"] = "7a2"
englishName["خ"] = "5hi"
englishName["د"] = "dal"
englishName["ذ"] = "zal"
englishName["ر"] = "ra2"
englishName["ز"] = "zeen"
englishName["س"] = "seen"
englishName["ش"] = "sheen"
englishName["ص"] = "sad"
englishName["ض"] = "dad"
englishName["ط"] = "ta2"
englishName["ظ"] = "za2"
englishName["ع"] = "3een"
englishName["غ"] = "5een"
englishName["ق"] = "2af"
englishName["ف"] = "fa2"
englishName["ك"] = "kaf"
englishName["ل"] = "lam"
englishName["م"] = "meem"
englishName["ن"] = "noon"
englishName["ه"] = "ha2"
englishName["و"] = "wow"
englishName["ي"] = "ya2"
englishName["ة"] = "ta2Marbota"
englishName["ئ"] = "ya2Hamza"
englishName["ؤ"] = "wowHamze"
englishName["ى"] = "alfLayna"
englishName["ء"] = "hamza"
englishName["لا"] = "lam2lf"
englishName["لأ"] = "lam2lfHamzafo2"
englishName["لإ"] = "lam2lfHamzaTa7t"
englishName["لآ"] = "lam2lfHamzaMada"

englishName["؟"] = "questionMark"
englishName["."] = 'dot'
englishName[","] = 'fasla1'
englishName['،'] = "fasla2"

englishName['1'] = 'one'
englishName['2'] = 'two'
englishName['3'] = 'three'
englishName['4'] = 'four'
englishName['5'] = 'five'
englishName['6'] = 'six'
englishName['7'] = 'seven'
englishName['8'] = 'eight'
englishName['9'] = 'nine'
englishName['0'] = 'zero'
englishName['('] = '('
englishName[')'] = ')'



nonConnChOneSide = ["و", "ر", "ز", "ذ", "د",
                    "ا", "أ", "إ", "ة", "ء", "؟", ".", ",", '،', "لأ", "لإ","لا","(",")"]
nonConnChTwoSide = ["ء", "؟", ".", ",", '،',"(",")"]




def check_on_length(wrong_file,right_file,ending=2000):
    fWrong = open(wrong_file,"w")
    fRight = open(right_file, "w")
    total=0
    wrong = 0
    for file in os.listdir("scanned//"):
        file=file.replace(".png","")
        img = cv2.imread("scanned\\"+file+".png")
        ftext = open("text\\"+file+".txt", "r", encoding="utf-8")
        ## image segmentation
        thre = binarize(img)
        rotated = textSkewCorrection(thre)
        wordList = wordSegmentation(rotated)

        ## text preprocessing
        text = ftext.read().replace(":", "").replace(".", "")

        ## tokenizations
        wordsArray = nltk.word_tokenize(text)

        
        total+=1
        #print("image Number ",total)
        #print(len(wordList), len (wordsArray))
        if len(wordList) != len (wordsArray):
            wrong +=1
            stt="{0:<20}  {1:<20}  {2:<20}".format(
                file, str(len(wordList)), str(len(wordsArray)))
            fWrong.write(stt+"\n")
            #print("************wrong Image************")
        else:
            fRight.write(file+"\n")
        
        ftext.close()
        if total == ending:
            break
    
    fWrong.close()
    fRight.close()




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



def createDataSet(img,wordList,wordsArray,characterDict,nonConnChOneSide,nonConnChTwoSide,englishName,file):

    c=0
    #print("lengthes ",len(wordList),"  ",len(wordsArray))
    for Wordimage,word in zip(wordList, wordsArray):
        rows=Wordimage["rows"]
        columns = Wordimage["columns"]

        image = img[rows[0]:rows[1], columns[0]:columns[1]]
        width=image.shape[1]

        #print(word,"xxx")
        #cv2.imshow("Word",image)
        #cv2.waitKey(0)
        i=0
        while i < len(word):
            #print(word[i])
            length=0
            char=""
            if (checkLamAlf(word,i)):
                length = characterDict[word[i:i+2]][0]
                char = word[i:i+2]
                i=i+1
            else:
                length=getCharacterLength(characterDict,nonConnChOneSide,nonConnChTwoSide,word,i)-1
                char=word[i]
            index= max(width-length, 0)
            charImage=image[:,index:width]

            if (charImage.shape[1] == 0):
                #print("something wrong in alingment")
                return
            #cv2.imshow("char", charImage)
            #cv2.waitKey(0)
            print('dataset\\'+ englishName[char]+"\\"+file)
            if not os.path.exists('dataset\\'+englishName[char]+"\\"+file):
                os.makedirs('dataset\\'+englishName[char]+"\\"+file)

            fileName = 'dataset\\' +englishName[char]+"\\"+file+"\\" + str(c)+".png"
            #print(fileName)
            cv2.imwrite(fileName, charImage)
            image = image[:, 0:max(width-length, 0)]
            width = image.shape[1]
            c+=1
            i=i+1

check_on_length("wrong.txt","right.txt")


#f = open("right.txt", "r")
#fw = open("output.txt", "w", encoding="utf-8")
#files = f.read().split()

#for file in files:

#        img = cv2.imread("scanned\\"+file+".png")
#        ftext = open("text\\"+file+".txt", "r", encoding="utf-8")
#        ## image segmentation

        ## text preprocessing
 #       text = ftext.read().replace(":", "").replace(".", "").replace("-","")

        ## tokenizations
#        wordsArray = nltk.word_tokenize(text)
#        for word in wordsArray:
#            fw.writelines([word, "  ", str(word.encode("utf8-")), "\n"])

#        thre = thresoldOtsu(img)
#        rotated = textSkewCorrection(thre)

        #cv2.imwrite("corr_capr1003.png", rotated)
#        wordList = wordSegmentation(rotated)

#        createDataSet(rotated,wordList, wordsArray, characterDict,
#                      nonConnChOneSide, nonConnChTwoSide, englishName, file)
        

#f.close()
#fw.close()


#check_on_length()


            













                



