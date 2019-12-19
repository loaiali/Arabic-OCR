########################### global ##############
datasetDir = "dataset"
# featuresDir = "dataset_features"
featuresDir = "raw_features"

############################################# Tuning #########################################################################
startIndex_tune = 11 # update it if you comment some configs
configsToRun = [3, 4, 5, 6, 8, 9, 10, 103, 102]
config_tune = [
    # N = 1 to 10
    {
        'N': 1,
        'gamma': 'auto',
        'C': 5
    },
    {
        'N': 3,
        'gamma': 'auto',
        'C': 50
    },
    {
        'N': 4,
        'gamma': 1,
        'C': 80
    },
    {
        'N': 5,
        'gamma': 30,
        'C': 80
    },
    {
        'N': 6,
        'gamma': 80,
        'C': 80
    },
    {
        'N': 7,
        'gamma': 50,
        'C': 10
    },
    {
        'N': 8,
        'gamma': 'auto',
        'C': 120,
        'kernel': 'linear'
    },
    {
        'N': 9,
        'gamma': 'auto',
        'C': 90,
        'kernel': 'poly'
    },
    {
        'N': 10,
        'gamma': 'auto',
        'C': 90,
        'kernel': 'sigmoid'
    },

    # N = 10 to 16
    {
        'N': 100,
        'gamma': 'auto',
        'C': 70
    },
    {
        'N': 101,
        'gamma': 'auto',
        'C': 80
    },
    {
        'N': 102,
        'gamma': 'auto',
        'C': 100
    },
    {
        'N': 103,
        'gamma': 'auto',
        'C': 130
    },
    {
        'N': 104,
        'gamma': 'auto',
        'C': 50
    },
    {
        'N': 105,
        'gamma': 'auto',
        'kernel': 'linear',
        'C': 70
    },
    {
        'N': 106,
        'gamma': 'auto',
        'kernel': 'linear',
        'C': 100
    },
]

############################################# Training and predicting ########################################################
currentTrainingConfig = {'gamma': 'auto', 'C': 80}
modelToPredict = "model.sav" # when you run predict.py, this is the model loaded
trainModelTo = "model_train_rawfeatures.sav" # after you run train.py, the model will be saved to this file

englishName = {}
englishName["ا"] = "alf"
englishName["ب"] = "ba2"
englishName["ت"] = "ta2"
englishName["ث"] = "tha2"
englishName["ج"] = "geem"
englishName["ح"] = "7a2"
englishName["خ"] = "5hi"
englishName["د"] = "dal"
englishName["ذ"] = "zal"
englishName["ر"] = "ra2"
englishName["ز"] = "zeen"
englishName["س"] = "seen"
englishName["ش"] = "sheen"
englishName["ص"] = "sad"
englishName["ض"] = "dad"
englishName["ط"] = "ta2"
englishName["ظ"] = "za2"
englishName["ع"] = "3een"
englishName["غ"] = "5een"
englishName["ق"] = "2af"
englishName["ف"] = "fa2"
englishName["ك"] = "kaf"
englishName["ل"] = "lam"
englishName["م"] = "meem"
englishName["ن"] = "noon"
englishName["ه"] = "ha2"
englishName["و"] = "wow"
englishName["ي"] = "ya2"
englishName["ة"] = "ta2Marbota"
englishName["ئ"] = "ya2Hamza"
englishName["ؤ"] = "wowHamze"
englishName["ى"] = "alfLayna"
englishName["لا"] = "lam2lf"
englishName["لأ"] = "lam2lfHamzafo2"
englishName["لإ"] = "lam2lfHamzaTa7t"
englishName["لآ"] = "lam2lfHamzaMada"

englishName["؟"] = "questionMark"
englishName["."] = 'dot'
englishName[","] = 'fasla1'
englishName['،'] = "fasla2"

englishName['1'] = 'one'
englishName['2'] = 'two'
englishName['3'] = 'three'
englishName['4'] = 'four'
englishName['5'] = 'five'
englishName['6'] = 'six'
englishName['7'] = 'seven'
englishName['8'] = 'eight'
englishName['9'] = 'nine'
englishName['0'] = 'zero'
englishName['('] = '('
englishName[')'] = ')'