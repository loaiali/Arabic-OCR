from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix

from trainalgos import train_svm as train
from config import config_tune as expectedConfigs, startIndex_tune as startIndex, datasetDir
from train import readFeaturesAndLabels
from ticktock import tick, tock


def writeToFile(i, config, mat, rep, score, clf):
    fid = f'N{i}_K-{config.get("kernel") or "rbf"}_g-{config.get("gamma") or "auto"}_C-{config.get("C") or "1"}'
    f = open(f'reports/{fid}.txt', 'w')
    f.write(f"config:\n{config}\n\nscore = {score}\n\nconfusion matrix:\n{mat}\n\nreport:\n{rep}\n\n")
    f.close()
    joblib.dump(clf, f"clfs/{fid}.sav")

def main():
    tick("loading features data from files")
    x_train, x_test, y_train, y_test = readFeaturesAndLabels(datasetDir)
    tock("data is loaded into python objects")

    tick("training different classifiers with different configurations")
    i = startIndex
    for config in expectedConfigs:
        tick(f"training the classifier with config {config}")
        currentClassifier = train(x_train, y_train, config)
        tock("classifier trained")

        tick("calculating the metrices")
        y_pred = currentClassifier.predict(x_test)
        mat = confusion_matrix(y_test, y_pred)
        rep = classification_report(y_test, y_pred)
        score = currentClassifier.score(x_test, y_test)
        tock("metrices calculated successfully")

        writeToFile(i, config, mat, rep, score, currentClassifier)
        i += 1
    tock("the whole test cases has ended")

if (__name__ == "__main__"):
    main()