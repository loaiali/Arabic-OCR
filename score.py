from sklearn.externals import joblib
from config import datasetDir

from train import readFeaturesAndLabels


clf = joblib.load("model_train_rawfeatures.sav")
x_test, y_test = joblib.load("features_test.sav"), joblib.load("labels_test.sav")
score = clf.score(x_test, y_test)
print(score)

# print([{i: label} for i, label in enumerate(clf.classes_)])

def main():
	x_train, x_test, y_train, y_test = readFeaturesAndLabels(datasetDir)

	clf = joblib.load("model_train_rawfeatures.sav")
	y_pred = clf.predict(x_test)
	score = clf.score(x_test, y_test)

	print(score)