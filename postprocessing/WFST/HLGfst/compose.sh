fstarcsort --sort_type="olabel"  H.fst  H_sorted.fst
fstarcsort --sort_type="ilabel" L.fst L_sorted.fst
fstarcsort --sort_type="ilabel" G.fst G_sorted.fst

echo 'compose H L G to HLG.fst'
fstcompose L_sorted.fst G_sorted.fst  | fstarcsort  --sort_type="olabel" > HL_sorted.fst
fstcompose HL_sorted.fst G_sorted.fst | fstprint > composed.txt


echo 'remove backoff terminate and disampig symbols form LG.txt'
sed -i 's/D[0-9]\+\|ـجـ\|ـأـ/٭/g' composed.txt


fstcompile --isymbols=input.syms --osymbols=output.syms --keep_osymbols --keep_isymbols < composed.txt | fstdeterminize | fstminimize | fstprint  > HLG_opt.txt;

# echo 'drawing HLG.fst and HLG_opt.fst'
# fstcompile --isymbols=input.syms --osymbols=output.syms --keep_osymbols --keep_isymbols HLG_opt.txt | fstdraw  -portrait | dot -Tpdf > HLG.pdf
# fstdraw  -portrait HLG_opt.fst | dot -Tpdf > HLG_opt.pdf

echo 'copy HLG_opt.txt to Decodingfolder'
cp HLG_opt.txt  ../../Decoding/HLG.txt