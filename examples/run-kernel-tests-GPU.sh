#/bin/bash
# source run-kernel-tests-GPU.sh > run-kernel-tests-GPU.out
for sz in $(cat oddsizes.txt)
do
   python test-mdrfsconv.py $sz 15 5 d GPU
   python test-mdrfsconv.py $sz 15 5 s GPU
   python test-stepphase.py $sz 15 5 d GPU
   python test-stepphase.py $sz 15 5 s GPU
done
