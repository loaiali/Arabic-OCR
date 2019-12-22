from sklearn.externals import joblib
import numpy as np
from config import modelToPredict as modelPath
from sys import argv
from featureExtraction import imgToFeatureVector, imageToFeatureVector
from sklearn.metrics import classification_report
import ticktock

arabic_letters_probs = [
    0.16464289, 0.05567354, 0.04702837, 0.00968891, 0.01483219, 0.07832473,
    0.03759651, 0.03138544, 0.11820532, 0.07037656, 0.06028702, 0.03778822,
    0.0081729, 0.03170526, 0.04741351, 0.02525668, 0.02020331, 0.02812318,
    0.01908568, 0.01025082, 0.00697089, 0.00247327, 0.02374559, 0.01513691,
    0.00620304, 0.00371042, 0.00985217, 0.00923583, 0.00663082
]
from scipy.special import softmax
def activationFunction(scores):
    return np.log(softmax(scores)) - \
        np.log(arabic_letters_probs)


model = None
def predictFromFeatureVector(xtest, withAllScores = False):
    global model
    if (model is None):
        ticktock.tick("loading the model")
        model = joblib.load(modelPath)
        ticktock.tock("model loaded")

    # print(model.score())
    if(not withAllScores):
        return model.predict([xtest])

    #! return list sorted every time
    # return model.decision_function([xtest])[0]
    # print(model.classes_[np.argmax(model.decision_function([xtest])[0])])
    #! return with dictionary for every class
    mapping = {label: score for label, score in zip(model.classes_, model.decision_function([xtest])[0])}
    return mapping

def predictFromImagePath(path, withAllScores = False):
    featureVector = imageToFeatureVector(path)
    labelOrMapping = predictFromFeatureVector(featureVector)
    return labelOrMapping

def predictFromImage(img, withAllScores = False):
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