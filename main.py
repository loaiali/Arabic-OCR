import sys
import numpy as np
import argparse
import os
from sklearn.externals import joblib
from config import modelToPredict
from segmentation2 import segmentationFromPath, showScaled
from predict import predictFromImage, activationFunction
from postprocessing.Decoding.fst import FST
from postprocessing.Decoding.beam_search import BeamSearch
from config import englishName, arabicNames
from time import time
# sys.path.append("postprocessing/Decoding")
'''
search graph params interface
[
    alfScore    ba2Score    ....    ya2Scode
]
'''



class OCR:
    def __init__(self, decodingGraphPath, inputLabelsPath, beamWidth,lmWeight,sentLen,withSearch):
        self.withSeach = withSearch
        if(withSearch):
            self.lmWeight = lmWeight
            self.beamWidth = beamWidth
            self.sentLen = sentLen
            self.fst = FST(decodingGraphPath, inputLabelsPath)
        self.indexToLetters = None
        with open(inputLabelsPath,"r", encoding="utf8") as f:
            indexToLetters = {i: englishName[letter] for (i,letter) in enumerate(f.read().split())}
                
        
    def getTextFromImage(self, imagePath):
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
        wordsSegmented, img = segmentationFromPath(imagePath)
        # joblib.dump(wordsSegmented, "wordsSegmented.test")
        # joblib.dump(img, "img.test")
        # wordsSegmented, img = joblib.load("wordsSegmented.test"), joblib.load("img.test")

        # [      {'rows': (), 'columns': (), 'srl': [{mid: int}]}  ,       {}, {}]
        wordsCount = 0
        letters = []
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

                if(self.withSeach):
                    scoresSorted = [lettersToScores[indexToLetters[i]] for i in range(len(indexToLetters.keys()))]
                    scoresSorted = activationFunction(scoresSorted)
                    letters.append(np.array(scoresSorted))
                else:
                    predLetter = arabicNames[sorted(lettersToScores.items(), key=lambda x: x[1])[-1][0]]
                    letters.append(predLetter) 

            wordsCount+=1
            if (self.withSeach):
                if(wordsCount >= self.sentLen):
                    predictedWords = self._search(np.array(letters))
                    allImageWords.append(''.join(predictedWords))
                    letters.clear()
            else:
                allImageWords.append(''.join(letters))
                letters.clear()
            
                
        return ' '.join(allImageWords)


    def _search(self, predMatrix):
        words = self.fst.decode(BeamSearch(self.beamWidth), predMatrix, self.lmWeight)
        return words


def main():
    parser = argparse.ArgumentParser(
        description="OCR parameters ")
    parser.add_argument('-search', '--search',
                        help='Enable serach or not', required=True, type=bool, default=True)
    parser.add_argument('-graph', '--graph',
                        help="Text-format openfst decoding graph", required=False, default='LG.txt')
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score',
                        required=False, type=float, default=1)
    parser.add_argument('-beam_width', '--beam_width',
                        help='Maximum token count per frame', required=False, type=int, default=250)
    parser.add_argument('-sentLen', '--sentLen',
                        help='Number of words in a sentence given to search', required=True, type=int, default=1)
    parser.add_argument('-ilabels', '--ilabels',
                        help="Text files containing input labels",type=str, required=True, default="input_labels.txt")
    parser.add_argument('-refPath', '--refPath',
                        help="Folder continaing refernces text files which are also image files names to run OCR on it",
                        type=str, required=True, default=None)
    parser.add_argument(
        '-predPath', '--predPath', help='path to write output hypotheses', type=str, required=True, default=None)
    parser.add_argument(
        '-imgsPath', '--imgsPath', help='Path where scanned images live',type=str, required=False, default='./scanned/')

    args = parser.parse_args()

    prog = OCR(args.graph, args.ilabels,lmWeight=args.lmweight, beamWidth= args.beam_width,sentLen=args.sentLen,withSearch=args.search)

    for fileName in os.listdir(args.refPath):
        startTime = time()
        print("Start image "+ fileName)

        predictedText = prog.getTextFromImage(os.path.join(args.imgsPath, fileName))

        print(f'Image {fileName} took {int(time=startTime)} seconds')

        with open(os.path.join(args.predPath, fileName),encoding="utf-8") as f:
            f.write(predictedText)

# python main.py -search 0 -graph LG0.txt -lmweight 1 -beam_width 250 -sentLen 1 -ilabels input_labels.txt -imgsPath ./scanned -refPath ./reference/ -predPath ./predicted
if __name__ == "__main__":
    main()