#/bin/bash
# source run-FFT-tests-CPU.sh > run-FFT-tests-CPU.out
for sz in $(cat fftsizes.txt)
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
