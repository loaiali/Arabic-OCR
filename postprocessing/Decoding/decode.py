from fst import FST
from classifier import Classifier
from beam_search import *
import os.path
import argparse
import time
import collections
import sys

epsSym = '٭'
startSym = 'ـسـ'
endSym = 'ـأـ'
backOffSym = 'ـجـ'

#  python3 decode.py -model / -decoding_graph LG.txt -input_labels ../input_labels -out try


def main():
    parser = argparse.ArgumentParser(
        description="Decode speech from parameter files.")
    parser.add_argument(
        '-model', '--model', help='classifier file', required=True, default=None)
    parser.add_argument('-decoding_graph', '--decoding_graph',
                        help="Text-format openfst decoding graph", required=True, default=None)
    parser.add_argument('-input_labels', '--input_labels',
                        help="Text files containing input labels", required=True, default=None)

    parser.add_argument('-testfile', '--testfile',
                        help="Text files containing sentences to test on",
                        type=argparse.FileType(mode='r', encoding='utf-8'), required=True, default=None)
    parser.add_argument(
        '-outfile', '--outfile', help='Filename to write output hypotheses', type=argparse.FileType(mode='w', encoding='utf-8'), required=True, default=None)
    parser.add_argument('-lmweight', '--lmweight', help='Relative weight of LM score',
                        required=False, type=float, default=1)
    parser.add_argument('-beam_width', '--beam_width',
                        help='Maximum token count per frame', required=False, type=int, default=500)
    args = parser.parse_args()

    classifier = Classifier()
    classifier.load_model(args.model)

    fst = FST(args.decoding_graph, args.input_labels)

    predictedSentences = []
    all_time_start = time.time()
    try:
        for i, sentence in enumerate(args.testfile.read().splitlines()):
            print(f"passed: {i} sentences")
            time_start = time.time()
            classifier.load_parameters(sentence)
            # classifier.stack_features()

            activations = classifier.eval()

            words = fst.decode(
                BeamSearch(args.beam_width), activations, lmweight=args.lmweight)

            predictedSentences.append(' '.join(words))
            print(
                f"{len(sentence)} letters takes {int(time.time()-time_start)} seconds")
    except KeyboardInterrupt:
        print("[CTRL+C detected]")
        text = '\n'.join(predictedSentences)
        args.outfile.write(text)  # remove firlst endline

    text = '\n'.join(predictedSentences)
    args.outfile.write(text)  # remove firlst endline
    print(f"total time  takes {int(time.time()-all_time_start)} seconds")


if __name__ == '__main__':
    main()
