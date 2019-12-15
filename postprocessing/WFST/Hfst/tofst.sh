python3 lmToFst.py -lmfile arabic_letters.bo -out_dir ./H

echo 'compile H.fst'
python3 removeSymbols.py -fst H.txt -i input.syms
fstcompile --isymbols=input.syms --osymbols=input.syms -keep_osymbols --keep_isymbols  < H.txt > H.fst;

echo 'optmize H.fst'
fstdeterminize H.fst | fstminimize | fstprint > H.txt

cp H.txt ../../Decoding/H.txt
