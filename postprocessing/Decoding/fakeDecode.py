import os.path
import re
from random import randrange, random
import numpy as np
spaceSym = 'ـ'


def fakeDecode(activations, inputLabales):
    arabic_letters = []
    with open(inputLabales, encoding="utf-8") as f:
        arabic_letters = np.array(list(filter(len, f.read().split())))
    return ''.join(arabic_letters[np.argmax(activations, axis=1)]).split(spaceSym)
