#/bin/bash
# source run-kernel-tests-CPU.sh > run-kernel-tests-CPU.out
for sz in $(cat oddsizes.txt)
do
   python test-mdrfsconv.py $sz 15 5 d CPU
   python test-mdrfsconv.py $sz 15 5 s CPU
   python test-stepphase.py $sz 15 5 d CPU
   python test-stepphase.py $sz 15 5 s CPU
done
