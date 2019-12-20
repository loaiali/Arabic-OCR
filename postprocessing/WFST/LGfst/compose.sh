fstarcsort --sort_type="olabel"  L.fst  L_sorted.fst
fstarcsort --sort_type="ilabel" G.fst G_sorted.fst

echo 'compose L and G to LG.fst'
fstcompose L_sorted.fst G_sorted.fst | fstprint > composed.txt

echo 'remove backoff terminate and disampig symbols form LG.txt'
sed -i 's/D[0-9]\+\|ـجـ\|ـأـ/٭/g' composed.txt

fstcompile --isymbols=input.syms --osymbols=output.syms --keep_osymbols --keep_isymbols < composed.txt | fstdeterminize | fstminimize >  LG.fst;

echo 'copy LG_opt.txt to Decodingfolder'
fstprint LG.fst > ../../Decoding/LG.txt