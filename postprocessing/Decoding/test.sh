shuf -n 10 ../WFST/arabic.norm.txt  > test.txt 

beam_width=$1
lmweight=$2
echo "Testing with  and beam width $beam_width lmwight $lmweight"

echo "Testing with LG"
python3 decode.py -model / -decoding_graph LG.txt -input_labels input_labels -outfile hyps.txt -test test.txt -beam_width $beam_width -lmweight $lmweight
python3 performance.py -hyp hyps.txt -ref test.txt

echo "Testing with H"
sed "s/ /\n/g" test.txt > test2.txt
python3 decode.py -model / -decoding_graph H.txt -input_labels input_labels -outfile hyps2.txt -test test2.txt -beam_width $beam_width -lmweight $lmweight
sed -i "s/ //g" hyps2.txt
python3 performance.py -hyp hyps2.txt -ref test2.txt