import collections
import numpy as np
import os.path
import re
import sys
from random import randrange, random

additional_letters = "لا|لأ|لآ|لإ"
arabic_letter = re.compile(f'{additional_letters}|[\u0621-\u064A]')

arabic_letters_probs = [0.15704748, 0.08990748, 0.00692419, 0.05519331, 0.08611871,
                        0.02410659, 0.04612263, 0.0519825, 0.03162658, 0.04284779,
                        0.02999402, 0.06160681, 0.07197932, 0.02626929, 0.05970513,
                        0.02277152, 0.00985534, 0.00466017, 0.01151384, 0.00237183,
                        0.02821879, 0.01281729, 0.01854503, 0.01683303, 0.0093017,
                        0.0109148, 0.00383984, 0.006925]


def getRandomEvals(inputFile="input_labels", text=""):
    removalErrRate = 1.
    subErrRate = .7
    arabic_letters = []
    with open(inputFile) as f:
        for line in f:
            arabic_letters.append(line.split()[0])
    arabic_letters = arabic_letters

    features_vectors = []

    for letter in re.findall(arabic_letter, text):
        if(random() > removalErrRate):
            continue

        features_vectors.append([randrange(1, 10)
                                 for letter in arabic_letters])
        curr_letter_indx = arabic_letters.index(letter)
        features_vectors[-1][curr_letter_indx] = 20. if random(
        ) < subErrRate else 1.
        from scipy.special import softmax
        features_vectors[-1] = softmax(features_vectors[-1])
        features_vectors[-1] = np.log(features_vectors[-1]) - \
            np.log(arabic_letters_probs)
        pritn(features_vectors[-1])
        # print(features_vectors[-1])

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
