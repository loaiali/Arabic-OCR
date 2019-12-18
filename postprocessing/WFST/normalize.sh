corpusFile=$1

#sentences
sed 's/[ؤإأآةءئى]//g;s/ و/ و /g;s/\:\|؟\|\;/\. /g; s/\.\+/\. /g; s/[^ء-يلا .]//g;s/ﻻ/لا/g' ./text/$corpusFile  > arabic.norm.txt
sed -i 's/\./\n/g' arabic.norm.txt
sed -i 's/ \+/ /g;/^\s*$/d' arabic.norm.txt

#words
sed 's/ /\n/g' arabic.norm.txt > arabic_letters.norm.txt
sed -i '/^\s*$/d;' arabic_letters.norm.txt
sed -i "s/\(.\)/\1 /g" arabic_letters.norm.txt

echo "Normalization Done"
