from sklearn.externals import joblib
from config import modelToPredict
from segmentation2 import segmentationFromPath, showScaled
from predict import predictFromImage, activationFunction
import sys
import numpy as np
# sys.path.append("postprocessing/Decoding")
from postprocessing.Decoding.fst import FST
from postprocessing.Decoding.beam_search import BeamSearch
from config import englishName
'''
search graph params interface
[
    alfScore    ba2Score    ....    ya2Scode
]
'''



class OCR:
    def __init__(self, decodingGraphPath, inputLabelsPath, lmWeight, beamWidth):
        self.fst = FST(decodingGraphPath, inputLabelsPath)
        self.lmWeight = lmWeight
        self.beamWidth = beamWidth
        
    def getWordsFromImage(self, imagePath):
        '''
            returns list of:
            [
                alfScore    ba2Score    ....    ya2Scode
                alfScore    ba2Score    ....    ya2Scode
                alfScore    ba2Score    ....    ya2Scode
                alfScore    ba2Score    ....    ya2Scode
                .
                .
                . * 15 times
            ]
        '''
        allImageWords = []
        # wordsSegmented, img = segmentationFromPath(imagePath)
        # joblib.dump(wordsSegmented, "wordsSegmented.test")
        # joblib.dump(img, "img.test")
        wordsSegmented, img = joblib.load("wordsSegmented.test"), joblib.load("img.test")

        words15 = []
        wordsCount = 0
        # [      {'rows': (), 'columns': (), 'srl': [{mid: int}]}  ,       {}, {}]
        for wordDictionary in wordsSegmented:
            x1, x2, y1, y2 = *wordDictionary['rows'], *wordDictionary['columns']
            currentWordImage = img[x1:x2, y1:y2]
            # showScaled(currentWordImage, "currentWordImage", 100)
            # loop through chars
            srl = wordDictionary['srl']
            srl.insert(0, {'mid': currentWordImage.shape[1]})
            srl.append({'mid': 0})
            for i in range(len(srl) - 1):
                midii = srl[i+1]['mid']
                midi = srl[i]['mid']
                currentCharImage = currentWordImage[:,midii:midi]
                # showScaled(currentCharImage, "currentCharImage", 100)
                lettersToScores = predictFromImage(currentCharImage, True)
                indexToLetters = None
                with open("input_labels.txt","r", encoding="utf8") as f:
                    indexToLetters = {i: englishName[letter] for (i,letter) in enumerate(f.read().split())}
                scoresSorted = [lettersToScores[indexToLetters[i]] for i in range(len(indexToLetters.keys()))]
                # print(scoresSorted)
                scoresSorted = activationFunction(scoresSorted)
                # print(scoresSorted)
                words15.append(np.array(scoresSorted))
            wordsCount += 1
            if (wordsCount >= 20):
                predictedWords = self._search(np.array(words15))
                print(predictedWords)
                # yield predictedWords
                allImageWords.append(predictedWords)
                wordsCount = 0
                words15.clear()
            # showScaled(currentCharImage, 'currentChar', 150)
        return ' '.join(allImageWords)


    def _search(self, predMatrix):
        print("searching:")
        words = self.fst.decode(BeamSearch(self.beamWidth), predMatrix, self.lmWeight)
        return words


def main():
    prog = OCR("LG4.txt", "input_labels.txt", 25, 250)
    imagePath = "scanned/capr1.png"
    words = prog.getWordsFromImage(imagePath)
    print(words)


if __name__ == "__main__":
    main()