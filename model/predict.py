from sklearn.externals import joblib
import numpy as np
from config import modelToPredict as modelPath
from sys import argv
from featureExtraction import imgToFeatureVector, imageToFeatureVector
from sklearn.metrics import classification_report


model = None


def predictFromFeatureVector(xtest, withAllScores=False):
    global model
    if (model is None):
        print(modelPath)
        model = joblib.load(modelPath)

    if(not withAllScores):
        return model.predict([xtest])


    scores = model.predict_log_proba([xtest])[0]

    mapping = {label: score for label, score in zip(
        model.classes_, scores)}
    return mapping


def predictFromImagePath(path, withAllScores=False):
    featureVector = imageToFeatureVector(path)
    labelOrMapping = predictFromFeatureVector(featureVector)
    return labelOrMapping


def predictFromImage(img, withAllScores=False):
    featureVector = imgToFeatureVector(img)
    labelOrMapping = predictFromFeatureVector(featureVector, withAllScores)
    return labelOrMapping


def main():
    if (len(argv) < 2):
        print("Error: you have to pass the image path")
        return
    label = predictFromImagePath(argv[1])
    print(label)


if __name__ == "__main__":
    main()
