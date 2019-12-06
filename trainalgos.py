from sklearn import svm


def train_svm(features, labels, config = None):
    defaultConfig = {
        # you can put your own default values here
        'gamma': 'auto',
        'C': 1.0
    }
    config = {**defaultConfig, **config}
    classifier = svm.SVC(**config)
    classifier.fit(features, labels)
    return classifier