import re
import string
import sys
import argparse

disampg = "D"
epsSym = '٭'
startSym = 'ـسـ'
endSym = 'ـأـ'
terminateSym = 'ـتـ'
spaceSym = 'ـ'
backOffSym = 'ـجـ'


arabic_word = re.compile('[\u0621-\u064A]+')
additional_letters = "لا|لأ|لآ|لإ"
arabic_letter = re.compile(f'{additional_letters}|[\u0621-\u064A]')
# disampg_char = re.compile(f'{disampg}/d+')
reservedDisampig = ["D0", "D00"]

arabicVocabStart = 4


def getLexicons(words):
    lexicons = {}
    for word in words[arabicVocabStart:]:
        letters = re.findall(arabic_letter, word)
        if(len(letters)):
            lexicons[word] = letters
    return lexicons


def addDisampigInputSyms(disampg, disampgKey):
    with open('input.syms', 'r') as f:
        lines = f.readlines()

    lines[-1] = lines[-1].replace('\n', '')
    num = len(lines)
    for i in range(disampgKey):
        lines.append(f'\n{disampg+str(i+1)} {int(num)}')
        num += 1

    for rd in reservedDisampig:
        lines.append(f'\n{rd} {int(num)}')
        num += 1

    with open('input.syms', 'w') as f:
        f.writelines(lines)


def resolveDisambiquity(lexicons):
    disampgKey = 1

    words = sorted(lexicons.keys())
    for i, word in enumerate(words):
        if i+1 < len(words) and words[i] == words[i+1][:len(word)]:
            lexicons[word] = lexicons[word] + [disampg+str((disampgKey+1))]
            disampgKey += 1

    addDisampigInputSyms(disampg, disampgKey)

    return lexicons


def lexiconToFst(lexicons):
    fst = [
        f'{0} {1} {startSym} {startSym}',
        f'{1} {1} {backOffSym} {backOffSym}',
        f'{1} {2} {spaceSym} {startSym}',  # may be restart a new sentnece
        f'{2} {1} {reservedDisampig[0]} {epsSym}',
        f'{1} {3} {spaceSym} {epsSym}',  # seprate between words
        f'{3} {1} {reservedDisampig[1]} {epsSym}',
    ]
    intial_state = 1
    avail_state = 4
    for word, letters in lexicons.items():
        for i, letter in enumerate(letters):
            src, dst, inp, out = (intial_state, avail_state, letter, epsSym)
            if(len(letters) == 1):
                dst = intial_state
                out = word
            elif(i == 0):
                dst = avail_state
                out = word
            elif(i == len(letters)-1):
                src = avail_state
                dst = intial_state
                avail_state += 1
            else:
                src = avail_state
                dst = avail_state+1
                avail_state += 1

            fst.append(f'{src} {dst} {inp} {out}')

    end = avail_state
    final = end+1
    # alwys end with space
    fst.append(f'{intial_state} {end} {spaceSym} {endSym}')
    fst.append(f'{end} {final} {terminateSym} {epsSym}')
    fst.append(f'{final}')
    return fst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'),
                        help='vocabfile', required=True)
    args = parser.parse_args()
    words = args.infile.read().split('\n')

    lexicons = getLexicons(words)
    lexicons = resolveDisambiquity(lexicons)
    fst = lexiconToFst(lexicons)

    with open('L.txt', 'w') as f:
        f.writelines('\n'.join(fst))
