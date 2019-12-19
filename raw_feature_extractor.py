from glob import glob
import numpy as np
import cv2
from Preprocessing import binarize
from segmentation2 import showScaled
import os
class RawFeatureExtractor:
    def __init__(self, datasetPath="dataset", outputPath="raw_features"):
        self.datasetPath = datasetPath
        self.outPath = outputPath

    def extract(self):
        folders = glob(f'{self.datasetPath}\\*')
        for folder in folders:
            print(f"currently in dire: {folder}")
            for test_case in glob(folder + '/*'):
                for f in glob(test_case + '/*.png'):
                    outputFolder = self.outPath + f[7:-4] + ".txt"
                    # characterImage=preprocessImage(f)
                    img = cv2.imread(f)
                    img = cv2.resize(img, (28, 28))
                    img = binarize(img)
                    self.writeFeatureVector(outputFolder, img.flatten())


    def writeFeatureVector(self, outputFolder, featureVector):
        os.makedirs(os.path.dirname(outputFolder), exist_ok=True)
        np.savetxt(outputFolder, featureVector, fmt='%d')


if __name__ == "__main__":
    rfe = RawFeatureExtractor()
    rfe.extract()