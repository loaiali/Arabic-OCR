echo 'lm to fst'
python3 lmToFst.py -lmfile arabic.bo -out_dir ./G

echo 'compile G.fst'
fstcompile --isymbols=input.syms --osymbols=input.syms --keep_osymbols --keep_isymbols < G.txt  >  G.fst;

cp G.fst ../LGfst/G.fst
cp input.syms ../LGfst/output.syms

cp G.fst ../HLGfst/G.fst
cp input.syms ../HLGfst/output.syms