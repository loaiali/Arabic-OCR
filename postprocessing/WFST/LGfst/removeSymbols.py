import re
import argparse
import os

epsSym = '٭'
startSym = 'ـسـ'
target_symbols = []
additional_letters = "لا|لأ|لآ|لإ"
arabic_letter = re.compile(f'^{additional_letters}|[\u0621-\u064A]$')


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', type=argparse.FileType(mode='r', encoding='utf-8'),
                    help='input symbols file', required=True)
parser.add_argument('-fst', '--fst', type=argparse.FileType(mode='r', encoding='utf-8'),
                    help='fstfile', required=True)
args = parser.parse_args()


def get_target_syms(text):
    syms = [line.split()[0] for line in text.splitlines()]
    return list(filter(lambda sym: not(arabic_letter.match(sym) or sym == epsSym), syms))


def remove_symbols():
    target_symbols = get_target_syms(args.infile.read())
    target_symbols.remove(startSym)
    target_symbols.sort(key=lambda x: -len(x))
    fst_text = args.fst.read()
    sedInput = 's/' + '\|'.join(target_symbols) + '/' + epsSym + '/g'
    command = f"sed -i '{sedInput}' {args.fst.name}"
    os.system(command)


if __name__ == '__main__':
    remove_symbols()
