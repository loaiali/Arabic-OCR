# shuf -n 3 ../WFST/arabic.norm.txt  > test.txt 
#rm *hyps*
beam_width=$1
lmweight=$2
graph1=$3
graph2=$4

echo "Testing with  and beam width $beam_width lmwight $lmweight with words graph $graph1 and letters graph $graph2"

#sed -i '/^\s*$/d;' test.txt # remove empty lines

if [ $graph1 != "NONE" ]
then
    echo "Testing with LG"
    python3 decode.py -model / -decoding_graph $graph1 -input_labels input_labels -outfile $graph1\_hyps.txt -test test.txt -beam_width $beam_width -lmweight $lmweight
    python3 performance.py -hyp $graph1\_hyps.txt  -ref test.txt > $graph1\_performance.txt
fi

if [ $graph2 != "NONE" ]
then
    echo "Testing with H"
    sed "s/ /\n/g" test.txt > test2.txt
    sed -i 's/ \+/ /g;' test2.txt
    sed -i '/^\s*$/d;' test2.txt
    python3 decode.py -model / -decoding_graph $graph2 -input_labels input_labels -outfile $graph2\_hyps.txt -test test2.txt -beam_width $beam_width -lmweight 10
    sed -i "s/ //g" $graph2\_hyps.txt
    python3 performance.py -hyp $graph2\_hyps.txt -ref test2.txt > $graph2\_performance.txt
fi
