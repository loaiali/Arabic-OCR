from sklearn.externals import joblib
import numpy as np
from config import modelToPredict as modelPath
from sys import argv
from featureExtraction import imageToFeatureVector

model = None
def predictFromFeatureVector(xtest):
    global model
    if (model is None):
        model = joblib.load(modelPath)
    return model.predict([xtest])

def predictFromImage(path):
    featureVector = imageToFeatureVector(path)
    label = predictFromFeatureVector(featureVector)
    return label


def main():
    if (len(argv) < 2):
        print("Error: you have to pass the image path")
        return
    label = predictFromImage(argv[1])
    print(label)



if __name__ == "__main__":
    main()