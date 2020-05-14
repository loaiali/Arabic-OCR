from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix

from trainalgos import train_svm as train
from config import config_tune as expectedConfigs, datasetDir, configsToRun
from train import readFeaturesAndLabels
from ticktock import tick, tock
from glob import glob

def writeToFile(i, config, mat, rep, score, clf):
    fid = f'N{i}_K-{config.get("kernel") or "rbf"}_g-{config.get("gamma") or "auto"}_C-{config.get("C") or "1"}'
    f = open(f'reports/{fid}.txt', 'w')
    f.write(f"config:\n{config}\n\nscore = {score}\n\nconfusion matrix:\n{mat}\n\nreport:\n{rep}\n\n")
    f.close()

def main():
    tick("loading features data from files")
    x_train, x_test, y_train, y_test = readFeaturesAndLabels(datasetDir)
    tock("data is loaded into python objects")

    tick("training different classifiers with different configurations")
    # startIndex = 0
    for config in expectedConfigs:
        i = config['N']
        if (not (i in configsToRun)):
            print(f"ignoring index {i}..")
            continue
        isExist = len(list(glob(f"reports/N{i}*.txt"))) >= 1
        if (isExist):
            print(f"already trained, ignoring index {i}..")
            continue
            
        
        del config['N']
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

    tock("the whole test cases has ended")


import gc
if (__name__ == "__main__"):
    while(True):
        try:
            main()
        except:
            print("Error in main(), shaklo mem error bardo")
            gc.collect()
            continue
        break
    print("alf mabrook")