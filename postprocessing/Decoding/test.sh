# shuf -n 3 ../WFST/arabic.norm.txt  > test.txt 
#rm *hyps*
beam_width=$1
lmweight=$2
graph1=$3
graph2=$4
search=$5
ngram=$6

echo "Testing with beam width $beam_width lmwight $lmweight with words graph $graph1 with ngram $ngram and letters graph $graph2"

#sed -i '/^\s*$/d;' test.txt # remove empty lines

if [ $graph1 != "NONE" ]
then
    echo "Testing with LG"
    if ! [ -z "$6" ]
    then
        sed -e :a -e '$!N; s/\n/ /; ta' test.txt |  sed "s/ /\n/$ngram;P;D" | sed "s/^ //g"  > test$ngram.txt
    fi
    
    python3 decode.py -model / -decoding_graph $graph1 -input_labels input_labels -outfile $graph1.$beam_width.$lmweight\_hyps.txt -test test$ngram.txt -beam_width $beam_width -lmweight $lmweight -search $search

    sed -e :a -e '$!N; s/\n/ /; ta' $graph1.$beam_width.$lmweight\_hyps.txt > ./predicted/1.txt
    sed -e :a -e '$!N; s/\n/ /; ta' test$ngram.txt > ./reference/1.txt
    python3 edit.py ./predicted/ ./reference/
fi

if [ $graph2 != "NONE" ]
then
    echo "Testing with H"
    sed "s/ /\n/g" test.txt > test2.txt
    sed -i 's/ \+/ /g;' test2.txt
    sed -i '/^\s*$/d;' test2.txt
    python3 decode.py -model / -decoding_graph $graph2 -input_labels input_labels -outfile $graph2.$beam_width.$lmweight\_hyps.txt -test test2.txt -beam_width $beam_width -lmweight $lmweight -search $search
    sed -i "s/ //g" $graph2.$beam_width.$lmweight\_hyps.txt
    sed -e :a -e '$!N; s/\n/ /; ta' $graph2.$beam_width.$lmweight\_hyps.txt > ./predicted/1.txt
    sed -e :a -e '$!N; s/\n/ /; ta' test.txt > ./reference/1.txt
    python3 edit.py ./predicted/ ./reference/
fi
