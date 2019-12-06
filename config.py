########################### global ##############
datasetDir = "dataset"
featuresDir = "dataset_features"

############################################# Tuning #########################################################################
startIndex_tune = 1 # update it if you comment some configs
config_tune = [
    # N = 1 to 10
    {
        'gamma': 'auto',
        'C': 5
    },
    {
        'gamma': 'auto',
        'C': 10
    },
    {
        'gamma': 'auto',
        'C': 50
    },
    {
        'gamma': 1,
        'C': 1.0
    },
    {
        'gamma': 10,
        'C': 1.0
    },
    {
        'gamma': 50,
        'C': 1.0
    },
    {
        'gamma': 50,
        'C': 10
    },
    {
        'gamma': 'auto',
        'C': 1.0,
        'kernel': 'linear'
    },
    {
        'gamma': 'auto',
        'C': 1.0,
        'kernel': 'poly'
    },
    {
        'gamma': 'auto',
        'C': 1.0,
        'kernel': 'sigmoid'
    },

    # N = 11 to 17
    {
        'gamma': 'auto',
        'C': 70
    },
    {
        'gamma': 'auto',
        'C': 80
    },
    {
        'gamma': 'auto',
        'C': 100
    },
    {
        'gamma': 'auto',
        'C': 50
    },
    {
        'gamma': 'auto',
        'kernel': 'linear',
        'C': 70
    },
    {
        'gamma': 'auto',
        'kernel': 'linear',
        'C': 100
    },
]

############################################# Training and predicting ########################################################
currentTrainingConfig = {'gamma': 'auto', 'C': 80}
modelToPredict = "model.sav" # when you run predict.py, this is the model loaded
trainModelTo = "model_train.sav" # after you run train.py, the model will be saved to this file