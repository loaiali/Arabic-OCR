# shuf -n 3 ../WFST/arabic.norm.txt  > test.txt 
#rm *hyps*
beam_width=$1
lmweight=$2
graph1=$3
graph2=$4
fakeDecode=$5

echo "Testing with beam width $beam_width lmwight $lmweight with words graph $graph1 and letters graph $graph2"

#sed -i '/^\s*$/d;' test.txt # remove empty lines

if [ $graph1 != "NONE" ]
then
    echo "Testing with LG"
    python3 decode.py -model / -decoding_graph $graph1 -input_labels input_labels -outfile $graph1\_hyps.txt -test test.txt -beam_width $beam_width -lmweight $lmweight -search $fakeDecode
    # python3 performance.py -hyp $graph1\_hyps.txt  -ref test.txt > $graph1\_performance.txt
    # cat $graph1\_performance.txt

    sed -e :a -e '$!N; s/\n/ /; ta' $graph1\_hyps.txt > ./predicted/1.txt
    sed -e :a -e '$!N; s/\n/ /; ta' test.txt > ./reference/1.txt
    python3 edit.py ./predicted/ ./reference/
fi

if [ $graph2 != "NONE" ]
then
    echo "Testing with H"
    sed "s/ /\n/g" test.txt > test2.txt
    sed -i 's/ \+/ /g;' test2.txt
    sed -i '/^\s*$/d;' test2.txt
    python3 decode.py -model / -decoding_graph $graph2 -input_labels input_labels -outfile $graph2\_hyps.txt -test test2.txt -beam_width $beam_width -lmweight $lmweight -search $fakeDecode
    sed -i "s/ //g" $graph2\_hyps.txt
    # python3 performance.py -hyp $graph2\_hyps.txt -ref test2.txt > $graph2\_performance.txt
    # cat $graph2\_performance.txt
    sed -e :a -e '$!N; s/\n/ /; ta' $graph2\_hyps.txt > ./predicted/1.txt
    sed -e :a -e '$!N; s/\n/ /; ta' test2.txt > ./reference/1.txt
    python3 edit.py ./predicted/ ./reference/
fi
