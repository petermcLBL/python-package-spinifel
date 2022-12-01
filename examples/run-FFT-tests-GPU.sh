#/bin/bash
# source run-FFT-tests-GPU.sh > run-FFT-tests-GPU.out
for sz in $(cat fftsizes.txt)
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
