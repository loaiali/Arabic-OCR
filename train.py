import numpy as np
import os
from glob import glob
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import ntpath

from trainalgos import train_svm as train
from ticktock import tick, tock
from config import trainModelTo as modelPath, currentTrainingConfig as trainConfig, featuresDir

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def readData(dataDir):
    features = []
    labels = []
    folders = glob(f'{dataDir}\\*')
    i = 0
    # for folder in folders:
    #     print(f"currently in folder: {folder}")
    #     for subDir in glob(folder+'\*'):
    #         for f in glob(subDir+'\*.txt'):
    #             i += 1
    # print ("you have total files =", i)
    for folder in folders:
        label = path_leaf(folder)
        print(f"currently in label: {label}")
        for subDir in glob(folder+'\*'):    
            # print(f"currently in subDir: {subDir}")
            for f in glob(subDir+'\*.txt'):
                featureVector = np.loadtxt(f)
                features.append(featureVector)
                labels.append(label)
    return features, labels

def readFeaturesAndLabels(dataDir):
    '''
        return x_train, x_test, y_train, y_test
    '''
    features = None
    isFiles = lambda *paths: all([os.path.isfile(path) for path in paths])
    if (isFiles("features_train.sav", "features_test", "labels_train", "labels_test")):
        x_train, y_train = joblib.load("features_train.sav"), joblib.load("labels_train.sav")
        x_test, y_test = joblib.load("features_test.sav"), joblib.load("labels_test.sav")
        return x_train, x_test, y_train, y_test
    if (isFiles("features_all.sav", "labels_all.sav")):
        features = joblib.load("features_all.sav")
        labels = joblib.load("labels_all.sav")
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20)
        joblib.dump(x_train, "features_train.sav")
        joblib.dump(y_train, "labels_train.sav")
        joblib.dump(x_test, "features_test.sav")
        joblib.dump(y_test, "labels_test.sav")
        return x_train, x_test, y_train, y_test

    features, labels = readData(dataDir)
    joblib.dump(features, 'features_all.sav')
    joblib.dump(labels, 'labels_all.sav')
    return readFeaturesAndLabels(dataDir)


def main():
    tick("reading trained and test features vectors")
    x_train, x_test, y_train, y_test = readFeaturesAndLabels(featuresDir)
    tock("done")

    tick("converting to np arrays")
    x_train = np.array(x_train, dtype=np.int16)
    x_test = np.array(x_test, dtype=np.int16)
    tock("done")

    tick(f"training the svm classifier over x_train, y_train with config {trainConfig}")    
    classifier = train(x_train, y_train, trainConfig)
    tock("classifier trained")

    tick(f"saving classifier model into {modelPath}")
    joblib.dump(classifier, modelPath)
    tock("model saved")


if (__name__ == "__main__"):
    main()