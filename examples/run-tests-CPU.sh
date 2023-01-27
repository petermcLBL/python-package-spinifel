#/bin/bash
# source run-tests-CPU.sh > run-tests-CPU.out
for sz in $(cat evensizes.txt oddsizes.txt)
do
   python test-fftn.py $sz F 105 5 d CPU
   python test-fftn.py $sz I 105 5 d CPU
   python test-fftn.py $sz F 105 5 s CPU
   python test-fftn.py $sz I 105 5 s CPU
   python test-rfftn.py $sz F 105 5 d CPU
   python test-rfftn.py $sz I 105 5 d CPU
   python test-rfftn.py $sz F 105 5 s CPU
   python test-rfftn.py $sz I 105 5 s CPU
done
for sz in $(cat oddsizes.txt)
do
   python test-mdrfsconv.py $sz 105 5 d CPU
   python test-mdrfsconv.py $sz 105 5 s CPU
   python test-stepphase.py $sz 105 5 d CPU
   python test-stepphase.py $sz 105 5 s CPU
done
