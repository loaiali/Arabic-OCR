import argparse
from collections import OrderedDict

disampg = "D"
epsSym = '٭'
startSym = 'ـسـ'
endSym = 'ـأـ'
# terminateSym = 'ـتـ'
backOffSym = 'ـجـ'


def getOutSymsAndSetMssing(filename):
    missingVocab = [backOffSym, endSym, startSym, epsSym]
    lines = []
    with open(filename, 'r') as vocab:
        lines = vocab.readlines()
        mvs = list(map(lambda x: x+'\n', missingVocab))
        [lines.remove(mv)
         for mv in mvs if mv in lines]
        for mv in mvs:
            lines.insert(0, mv)
    # lines = list(OrderedDict.fromkeys(lines))

    with open(filename, 'w') as vocab:
        vocab.writelines(lines)

    with open(filename, 'r') as src:
        with open('output.syms', 'w') as dst:
            i = 0
            for line in src:
                line = line.split()[0]+' '+str(i)+'\n'
                dst.write(line)
                i += 1


def getInSymsAndSetMssing(filename):
    missingInputs = [backOffSym, endSym, startSym, epsSym]
    lines = []
    with open(filename, 'r') as inputs:
        lines = inputs.readlines()
        mis = list(map(lambda x: x+'\n', missingInputs))
        [lines.remove(mi)
         for mi in mis if mi in lines]
        for mi in mis:
            lines.insert(0, mi)

    with open(filename, 'w') as inputs:
        inputs.writelines([line.split()[0]+'\n' for line in lines])

    with open(filename, 'r') as src:
        with open('input.syms', 'w') as dst:
            i = 0
            for line in lines:
                line = line.split()[0]+' '+str(i)+'\n'
                dst.write(line)
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'),
                        help='lettersfile', required=True)
    parser.add_argument('-o', '--outfile', type=argparse.FileType(mode='r', encoding='utf-8'),
                        help='vocabfile', required=True)
    args = parser.parse_args()
    getInSymsAndSetMssing(args.infile.name)
    getOutSymsAndSetMssing(args.outfile.name)
