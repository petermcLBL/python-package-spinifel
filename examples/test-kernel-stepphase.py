#! python

# Usage:
# python test-stepphase.py [N] [d|f] [GPU|CPU]
# N: size, default 81
# d|f: double (default) or float
# GPU|CPU: run on GPU (default) or on CPU
# 

import sys
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import fftx

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("python test-stepphase-mine.py sz [ d|s  [ GPU|CPU ]]")
    print("  sz = N")
    print("  d = double (default), s = single precision")
    sys.exit()

#Cube Size
N = 81
if len(sys.argv) > 1:
    N = int ( sys.argv[1] )

src_type = np.double
if len(sys.argv) > 2:
    if sys.argv[2] == "s":
        src_type = np.single

forGPU = (cp != None)
if len ( sys.argv ) > 3:
    if sys.argv[3] == "CPU":
        forGPU = False

strPU = "CPU"
if forGPU:
    strPU = "GPU"

print('Phasing kernel size ' + str(N) + ' ' + str(src_type) + ' on ' + strPU)

dims = [N, N, N]
dimsTuple = tuple(dims)

#build test input in numpy (cupy does not have itemset)
src = np.ones(dimsTuple, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k, np.random.random()*10.0)

xp = np
if forGPU:
    xp = cp
    #convert src to CuPy array
    src = cp.asarray(src)

#set amplitudes to a function of |fft(src)|
amplitudes = xp.absolute(xp.fft.rfftn(src))**3
amps_full = xp.absolute(xp.fft.fftn(src))**3

fftx_result = fftx.convo.stepphase(src, amplitudes)

#original spinifel calculation
rho_hat = xp.fft.fftn(src)
phases = xp.angle(rho_hat)
amp_mask = xp.ones(dims, dtype=xp.bool_)
amp_mask[0, 0, 0] = 0
rho_hat_mod = xp.where(amp_mask, amps_full*xp.exp(1j*phases), rho_hat)
spinifel_result = xp.fft.ifftn(rho_hat_mod).real

max_spinifel = xp.max(xp.absolute(spinifel_result))
max_diff = xp.max(xp.absolute(spinifel_result - fftx_result))
print ('Relative diff between spinifel and FFTX kernels =  ' + str(max_diff/max_spinifel) )
