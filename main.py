import sys
import numpy as np
import argparse
import os
from sklearn.externals import joblib
from config import modelToPredict
from segmentation4 import segmentationFromPath, showScaled
from predict import predictFromFeatureVector, activationFunction
from postprocessing.Decoding.fst import FST
from postprocessing.Decoding.beam_search import BeamSearch
from config import englishName, arabicNames
from time import time
from raw_feature_extractor import extractFeatures
import ticktock
# sys.path.append("postprocessing/Decoding")
'''
search graph params interface
[
    alfScore    ba2Score    ....    ya2Scode
]
'''


class OCR:
    def __init__(self, decodingGraphPath, inputLabelsPath, beamWidth, lmWeight, sentLen, withSearch, featureExtracor=extractFeatures):
        self.withSeach = withSearch
        if(withSearch):
            self.lmWeight = lmWeight
            self.beamWidth = beamWidth
            self.sentLen = sentLen
            self.fst = FST(decodingGraphPath, inputLabelsPath)
        self.indexToLetters = None
        with open(inputLabelsPath, "r", encoding="utf8") as f:
            self.indexToLetters = {i: englishName[letter] for (
                i, letter) in enumerate(f.read().split()[:-1])}
        self.featureExtracor = featureExtracor

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
        wordsSegmented, img = segmentationFromPath(imagePath, ticktock)
        # joblib.dump(wordsSegmented, "wordsSegmented.test")
        # joblib.dump(img, "img.test")
        # wordsSegmented, img = joblib.load("wordsSegmented.test"), joblib.load("img.test")

        # [      {'rows': (), 'columns': (), 'srl': [{mid: int}]}  ,       {}, {}]
        wordsCount = 0
        words = []
        letters = []
        lettersScores = []
        for wordDictionary in wordsSegmented:
            x1, x2, y1, y2 = * \
                wordDictionary['rows'], * wordDictionary['columns']
            currentWordImage = img[x1:x2, y1:y2]
            # showScaled(currentWordImage, "currentWordImage", 100)
            # loop through chars
            srl = wordDictionary['srl']
            srl.insert(0, {'mid': currentWordImage.shape[1]})
            srl.append({'mid': 0})
            letters = []
            lettersScores = []
            for i in range(len(srl) - 1):
                midii = srl[i+1]['mid']
                midi = srl[i]['mid']
                currentCharImage = currentWordImage[:, midii:midi]
                # showScaled(currentCharImage, "currentCharImage", 100)
                featureVector = self.featureExtracor(currentCharImage)
                lettersToScores = predictFromFeatureVector(
                    featureVector, withAllScores=True)
                # lettersToScores = predictFromFeatureVector(featureVector, True)

                if(self.withSeach):
                    scoresSorted = [lettersToScores[self.indexToLetters[i]]
                                    for i in range(len(self.indexToLetters))]
                    # scoresSorted = activationFunction(scoresSorted)
                    lettersScores.append(np.array(scoresSorted))
                else:
                    predLetter = arabicNames[sorted(
                        lettersToScores.items(), key=lambda x: x[1])[-1][0]]
                    letters.append(predLetter)

            wordsCount += 1
            if (self.withSeach):
                # if(self.sentLen >= 1):
                    # add space
                spaceScoresForPrevLetters = np.log(
                    0.0001)*np.ones((len(lettersScores), 1))
                lettersWithSpacesScores = np.hstack(
                    (lettersScores, spaceScoresForPrevLetters))

                spaceScoresForSpace = np.log(
                    .6/len(lettersWithSpacesScores[0])*np.ones(len(lettersWithSpacesScores[0])))
                spaceScoresForSpace[-1] = np.log(.4)

                if(len(words)):
                    words = np.vstack((words, lettersWithSpacesScores))
                else:
                    words = lettersWithSpacesScores
                words = np.vstack((words, spaceScoresForSpace))
                # else:
                #     words = np.array(lettersScores.copy())

                if(wordsCount == self.sentLen or wordDictionary == wordsSegmented[-1]):
                    # print(words.shape)
                    predictedWords = self._search(words)
                    # print(predictedWords)
                    # print(len(predictedWords))
                    allImageWords.append(' '.join(predictedWords))
                    words = []
                    wordsCount = 0
            else:
                allImageWords.append(''.join(letters))

        return ' '.join(allImageWords)

    def _search(self, predMatrix):
        words = self.fst.decode(BeamSearch(
            self.beamWidth), predMatrix, self.lmWeight)
        return words


def main():
    parser = argparse.ArgumentParser(
        description="OCR parameters ")
    parser.add_argument('-search', '--search',
                        help='Enable serach or not', required=True, type=str, default="False")
    parser.add_argument('-graph', '--graph',
                        help="Text-format openfst decoding graph", required=False, default='LG.txt')
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score',
                        required=False, type=float, default=1)
    parser.add_argument('-beam_width', '--beam_width',
                        help='Maximum token count per frame', required=False, type=int, default=250)
    parser.add_argument('-sentLen', '--sentLen',
                        help='Number of words in a sentence given to search', required=True, type=int, default=1)
    parser.add_argument('-ilabels', '--ilabels',
                        help="Text files containing input labels", type=str, required=True, default="input_labels.txt")
    # parser.add_argument('-refPath', '--refPath',
    #                     help="Folder continaing refernces text files which are also image files names to run OCR on it",
    #                     type=str, required=True, default=None)
    parser.add_argument(
        '-predPath', '--predPath', help='path to write output hypotheses', type=str, required=True, default=None)
    parser.add_argument(
        '-tp', '--timePath', help='path to write output time for each image', type=str, required=True, default=None)
    parser.add_argument(
        '-imgsPath', '--imgsPath', help='Path where scanned images live', type=str, required=False, default='./scanned/')

    args = parser.parse_args()

    withSearch = args.search == "True"
    prog = OCR(args.graph, args.ilabels, lmWeight=args.lmweight,
               beamWidth=args.beam_width, sentLen=args.sentLen, withSearch=withSearch)

    timeFile = open(args.timePath, 'w')
    for fileName in os.listdir(args.imgsPath):
        startTime = time()
        print("Start image " + fileName)

        predictedText = prog.getTextFromImage(args.imgsPath + "/" + fileName)
        elapsedSeconds = ticktock.tock("", log=False)

        print(f'Image {fileName} took {int(time()-startTime)} seconds')
        with open(os.path.join(args.predPath, fileName), 'w', encoding="utf-8") as f:
            f.write(predictedText)
        timeFile.write(str(elapsedSeconds) + '\n')
    timeFile.close()

# python3 main.py --timePath ./output/running_time.txt -search False -graph LG3g.txt -lmweight 1 -beam_width 250 -sentLen 1 -ilabels input_labels.txt -imgsPath ./scanned -predPath ./output/text/
if __name__ == "__main__":
    main()
