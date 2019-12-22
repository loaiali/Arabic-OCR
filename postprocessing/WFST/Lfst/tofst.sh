rm *.fst

python3 getInOutSyms.py -i arabic_letters -o vocab.vocab
cp output.syms ../Gfst/

python3 getLexiconFst.py -i vocab.vocab

fstcompile  --isymbols=input.syms --osymbols=output.syms  --keep_osymbols --keep_isymbols < L.txt > L.fst;

# fstdraw  -portrait L.fst | dot -Tpdf > L.pdf

cp L.fst ../LGfst/L.fst
cp L.fst ../HLGfst/L.fst
cp input.syms ../LGfst/
cp input.syms ../HLGfst/
cp input.syms ../Hfst/