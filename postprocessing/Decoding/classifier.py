import collections
import numpy as np
import os.path
import re
import sys
from random import randrange, random

additional_letters = "لا|لأ|لآ|لإ"
arabic_letter = re.compile(f'{additional_letters}|[\u0621-\u064A]')

arabic_letters_probs = [0.16464289, 0.05567354, 0.04702837, 0.00968891, 0.01483219,
                        0.07832473, 0.03759651, 0.03138544, 0.11820532, 0.07037656,
                        0.06028702, 0.03778822, 0.0081729, 0.03170526, 0.04741351,
                        0.02525668, 0.02020331, 0.02812318, 0.01908568, 0.01025082,
                        0.00697089, 0.00247327, 0.02374559, 0.01513691, 0.00620304,
                        0.00371042, 0.00985217, 0.00923583, 0.00663082]


def getRandomEvals(inputFile="input_labels", text=""):
    removalErrRate = 0.
    subErrRate = 0.
    arabic_letters = []
    with open(inputFile) as f:
        for line in f:
            arabic_letters.append(line.split()[0])
    arabic_letters = arabic_letters

    features_vectors = []

    for letter in re.findall(arabic_letter, text):
        if(random() < removalErrRate):
            continue
        features_vectors.append([randrange(1, 10)
                                 for letter in arabic_letters])
        curr_letter_indx = arabic_letters.index(letter)
        features_vectors[-1][curr_letter_indx] = randrange(20, 30) if random(
        ) > subErrRate else 1.
        from scipy.special import softmax
        features_vectors[-1] = softmax(features_vectors[-1])
        features_vectors[-1] = np.log(features_vectors[-1]) - \
            np.log(arabic_letters_probs)
    return features_vectors


class Classifier:
    def stack_features(self, context_frames=11):
        self.feature_vectors = np.column_stack([
            self.feature_vectors[np.minimum(len(self.feature_vectors) - 1, np.maximum(
                0, np.arange(len(self.feature_vectors), dtype=np.int) + d))]
            for d in range(-context_frames, context_frames + 1)
        ])

    def load_parameters(self, sentence="عندما عزل امير المؤمنين عمر بن الخطاب معاوية من بلاد الشام تلفت حواليه يبحث عن بديل يوليه مكانه"):
        self.text = sentence

    def load_model(self, *args):
        pass

    def eval(self):
        return np.array(getRandomEvals(text=self.text)).astype('f')


if __name__ == "__main__":
    c = Classifier()
    features = getRandomFeatures()
    print(features)
    print(min(features))
