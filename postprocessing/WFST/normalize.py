import re
import string
import sys
import os
import argparse
from math import ceil
import numpy as np

arab_word = re.compile('[\u0621-\u064A]+')
additional_letters = "ﻻ|لأ|لآ|لإ"
arabic_letter = re.compile(f'ﻻ|[\u0621-\u064A]')


def cutSentToWords(text, wordsLen):
    sentenceLen = 15
    words = text.split()
    padding = sentenceLen-len(words) % sentenceLen
    words += [' '] * padding
    sentences = np.reshape(
        words, (len(words)//sentenceLen, sentenceLen)).tolist()
    sentences = list(map(lambda x: ' '.join(x), sentences))
    sentences[-1] = str.replace(sentences[-1], ' ', '')  # remove padding
    return sentences


# def normalize_arabic(text):
#     # global allchars
#     sentences = text.split(". ")

#     def normalize(sentence):
#         return sentence
#         # arabic_words = re.findall(arab_word, sentence)
#         # arabic_words = [i for i in arabic_words if len(i) != 1 or i == u'و']
#         # return ' '.join(arabic_words)

#     normalized = filter(len, [normalize(sentence) for sentence in sentences])
#     return '\n'.join(normalized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'),
                        help='input file.', required=True)
    args = parser.parse_args()
    filetext = args.infile.read()
    # normalized = normalize_arabic(filetext)

    with open("arabic_letters.norm.txt", "w") as f:
        f.writelines('\n'.join(set([' '.join(re.findall(arabic_letter, word))
                                    for word in filetext.split()])))
