#/bin/bash
# source run-tests-GPU.sh > run-tests-GPU.out
for sz in $(cat evensizes.txt oddsizes.txt)
do
   python test-fftn.py $sz F 105 5 d GPU
   python test-fftn.py $sz I 105 5 d GPU
   python test-fftn.py $sz F 105 5 s GPU
   python test-fftn.py $sz I 105 5 s GPU
   python test-rfftn.py $sz F 105 5 d GPU
   python test-rfftn.py $sz I 105 5 d GPU
   python test-rfftn.py $sz F 105 5 s GPU
   python test-rfftn.py $sz I 105 5 s GPU
done
for sz in $(cat oddsizes.txt)
do
   python test-mdrconv.py $sz 105 5 d GPU
   python test-mdrconv.py $sz 105 5 s GPU
   python test-stepphase.py $sz 105 5 d GPU
   python test-stepphase.py $sz 105 5 s GPU
done
