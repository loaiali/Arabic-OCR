import numpy as np
import argparse


def sed(ref=None, hyp=None):
    if ref is None or hyp is None:
        RuntimeError("ref and hyp are required, cannot be None")

    x = ref
    y = hyp
    tokens = len(x)
    if (len(hyp) == 0):
        return (tokens, tokens, tokens, 0, 0)

    # p[ix,iy] consumed ix tokens from x, iy tokens from y
    p = np.PINF * np.ones((len(x) + 1, len(y) + 1))  # track total errors
    # track deletions, insertions, substitutions
    e = np.zeros((len(x)+1, len(y) + 1, 3), dtype=np.int)
    p[0] = 0
    for ix in range(len(x) + 1):
        for iy in range(len(y) + 1):
            cst = np.PINF*np.ones([3])
            s = 0
            if ix > 0:
                cst[0] = p[ix - 1, iy] + 1  # deletion cost
            if iy > 0:
                cst[1] = p[ix, iy - 1] + 1  # insertion cost
            if ix > 0 and iy > 0:
                s = (1 if x[ix - 1] != y[iy - 1] else 0)
                cst[2] = p[ix - 1, iy - 1] + s  # substitution cost
            if ix > 0 or iy > 0:
                idx = np.argmin(cst)  # if tied, one that occurs first wins
                p[ix, iy] = cst[idx]

                if (idx == 0):  # deletion
                    e[ix, iy, :] = e[ix - 1, iy, :]
                    e[ix, iy, 0] += 1
                elif (idx == 1):  # insertion
                    e[ix, iy, :] = e[ix, iy - 1, :]
                    e[ix, iy, 1] += 1
                elif (idx == 2):  # substitution
                    e[ix, iy, :] = e[ix - 1, iy - 1, :]
                    e[ix, iy, 2] += s

    edits = int(p[-1, -1])
    deletions, insertions, substitutions = e[-1, -1, :]
    return (tokens, edits, deletions, insertions, substitutions)


def score(reference: [str] = None, hypothesis: [str] = None):
    assert(len(hypothesis) == len(reference))
    wer = ser = total_tokens_n = 0
    for reference_string, hypothesis_string in zip(reference, hypothesis):
        reference_string_words = reference_string.split()
        hypothesis_string_words = hypothesis_string.split()
        tokens_n, errs_n, del_n, insert_n, sub_n = sed(
            ref=reference_string_words, hyp=hypothesis_string_words)
        wer += errs_n
        ser += 1 if errs_n > 0 else 0
        total_tokens_n += tokens_n
        # print(f'ref -> {reference_string}hyp-> {hypothesis_string}\nwords_n: {tokens_n}, sub_errs: {sub_n}, del_errs: {del_n}, insert_errs: {insert_n}, total_erros: {errs_n} \n')

    print(f'wer count: {wer}, ser count: {ser}')
    wer /= total_tokens_n
    ser /= len(hypothesis)
    print(f'wer% : {round(wer*100,2)}, ser% : {round(ser*100,2)}')
    return wer, ser


def read_trn_hyp_files(ref_trn=None, hyp_trn=None):
    with open(ref_trn) as reference:
        with open(hyp_trn) as hypothesis:
            return reference.readlines(), hypothesis.readlines()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument(
        '-hyp', '--hyp', type=argparse.FileType(mode='r', encoding='utf-8'), help='Hypothesized sentences', required=True, default=None)
    parser.add_argument(
        '-ref', '--ref', type=argparse.FileType(mode='r', encoding='utf-8'), help='Reference sentences', required=True, default=None)
    args = parser.parse_args()

    score(reference=args.ref.read().splitlines(),
          hypothesis=args.hyp.read().splitlines())
