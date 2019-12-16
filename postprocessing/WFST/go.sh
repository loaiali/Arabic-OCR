corpusFile=$1
vocabSize=$2
wordsOrder=$3
lettersOrder=$4
prunThreshold=$5
rm *.vocab
rm *.gz
rm *.bo
export LANG=ar_SY.UTF-8 LC_COLLATE=C.UTF-8 LANGUAGE=syr:ar:en;  unset LC_ALL;

echo corpusFile is $corpusFile , vocabSize is $vocabSize, ngram word order is $wordsOrder , ngram letters order is $lettersOrder  and pruning threshold $prunThreshold

#for file in `ls ./text | sort -R | head -$corpusSize`;
#;s/لا/لا/g
sed 's/[ؤإأآةءئى]//g;s/ و/ و /g;s/\:\|؟\|\;/\. /g; s/\.\+/\. /g; s/[^ء-يلا .]//g' ./text/$corpusFile  > arabic.norm.txt
sed -i 's/\./\n/g' arabic.norm.txt
sed -i 's/ \+/ /g;/^\s*$/d' arabic.norm.txt
# sed -i '/^\s*$/d;' arabic.norm.txt
sed 's/ /\n/g' arabic.norm.txt > arabic_letters.norm.txt
sed -i '/^\s*$/d;' arabic_letters.norm.txt
python3 normalize.py -i arabic_letters.norm.txt

ngram-count -text arabic.norm.txt -order 1 -write arabic.1grams
sort -k 2,2 -n -r arabic.1grams | head -$vocabSize > arabic.top$vocabSize.1grams
cut -f 1 arabic.top$vocabSize.1grams | sort > arabic.top$vocabSize.vocab
rm arabic.top$vocabSize.1grams

echo "OOV score for words"
compute-oov-rate arabic.top$vocabSize.vocab arabic.1grams

cat arabic.top$vocabSize.vocab | sed '/<s>/d' | sed '/<\/s>/d'  > ./Lfst/vocab.vocab
cp arabic_letters  ./Lfst/
cp arabic_letters ../Decoding/input_labels


echo 'preparing laguage mode for words'
ngram-count -text arabic.norm.txt -order ${wordsOrder} -write arabic.${wordsOrder}grams.gz
ngram-count -order ${wordsOrder} -vocab arabic.top$vocabSize.vocab -read arabic.${wordsOrder}grams.gz -wbdiscount -lm arabic.${wordsOrder}bo.gz

echo 'prune language model for words'
ngram  -lm arabic.${wordsOrder}bo.gz -prune $prunThreshold -write-lm arabic-pruned.${wordsOrder}bo.gz
rm arabic.${wordsOrder}bo.gz
 
gunzip arabic-pruned.${wordsOrder}bo.gz
cat arabic-pruned.${wordsOrder}bo | sed 's/<s>/ـسـ/g' | sed 's/<\/s>/ـأـ/g' > ./Gfst/arabic.bo
cp ./Gfst/arabic.bo ../Decoding
rm arabic-pruned.${wordsOrder}bo

#----------------------
echo 'preparing laguage mode for letters'
ngram-count -text arabic_letters.norm.txt -order ${lettersOrder} -write arabic_letters.${lettersOrder}grams.gz
ngram-count -order ${lettersOrder} -vocab arabic_letters -read arabic_letters.${lettersOrder}grams.gz -wbdiscount -lm arabic_letters.${lettersOrder}bo.gz

echo 'prune language model for letters'
ngram  -lm arabic_letters.${lettersOrder}bo.gz -prune $prunThreshold -write-lm arabic_letters-pruned.${lettersOrder}bo.gz
rm arabic_letters.${lettersOrder}bo.gz
 
gunzip arabic_letters-pruned.${lettersOrder}bo.gz
cat arabic_letters-pruned.${lettersOrder}bo | sed 's/<s>/ـسـ/g' | sed 's/<\/s>/ـأـ/g' > ./Hfst/arabic_letters.bo
rm arabic_letters-pruned.${lettersOrder}bo

echo 'produce lexicon model fst'
cd Lfst && ./tofst.sh && cd ..

echo 'produce language model fst'
cd Gfst && ./tofst.sh && cd ..

echo 'produce H fst'
cd Hfst && ./tofst.sh && cd ..

echo 'produce LG fst'
cd LGfst && ./compose.sh && cd ..

# echo 'produce HLG fst'
# cd HLGfst && ./compose.sh && cd ..

# echo '٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭٭'
