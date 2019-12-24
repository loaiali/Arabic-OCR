images=$1
text=$2
dest=$3

for file in `ls $images`; do cp $text`echo $file | cut -f1 -d'.'`.txt $dest;done
