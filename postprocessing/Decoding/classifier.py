import collections
import numpy as np
import os.path
import re
import sys
from random import randrange, random, choice

additional_letters = "لا|لأ|لآ|لإ"
spaceSym = 'ـ'
arabic_letter = re.compile(f'{additional_letters}|[\u0621-\u064A]')

# arabic_letters_probs = [0.16464289, 0.05567354, 0.04702837, 0.00968891, 0.01483219,
#                         0.07832473, 0.03759651, 0.03138544, 0.11820532, 0.07037656,
#                         0.06028702, 0.03778822, 0.0081729, 0.03170526, 0.04741351,
#                         0.02525668, 0.02020331, 0.02812318, 0.01908568, 0.01025082,
#                         0.00697089, 0.00247327, 0.02374559, 0.01513691, 0.00620304,
#                         0.00371042, 0.00985217, 0.00923583, 0.00663082]


def getRandomEvals(inputFile="input_labels", text=""):
    removalErrRate = .0
    subErrRate = .15
    insertionErrRate = .15
    arabic_letters = []
    with open(inputFile, encoding="utf-8") as f:
        arabic_letters = list(filter(len, f.read().split()))

    activations = []
    text = re.sub(" ", spaceSym, text)
    sent = re.findall(arabic_letter, text)
    for i in range(int(insertionErrRate*len(sent))):
        sent.insert(randrange(0, len(sent)), choice(arabic_letters[:-1]))
    if(sent[-1] != spaceSym):
        sent.append(spaceSym)

    # print(sent)
    removed = 0
    for letter in sent:
        if(letter == spaceSym):
            activations.append([.000001
                                for letter in arabic_letters])
            activations[-1][-1] = 1-sum(activations[-1])
            activations[-1] = np.log(activations[-1])
            continue

        if(len(sent)-removed > 1 and random() < removalErrRate):
            removed += 1
            continue
        activations.append([randrange(7, 10)
                            for letter in arabic_letters])
        curr_letter_indx = arabic_letters.index(letter)
        activations[-1][curr_letter_indx] = randrange(
            15, 18) if random() > subErrRate else randrange(7, 10)
        from scipy.special import softmax
        activations[-1] = np.sqrt(activations[-1])
        activations[-1] = softmax(activations[-1])
        activations[-1] = np.log(activations[-1])
        # np.savetxt("activations.txt", np.exp(activations), fmt='%5f')
    return activations


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
