#! python

# Usage:
# python test-kernel-mdrfsconv.py [N] [d|s] [GPU|CPU]
# N: size, default 81
# d|s: double (default) or single precision
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
    print("python test-kernel-mdrfsconv-mine.py sz [ d|s  [ GPU|CPU ]]")
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

print('Free-space convolution kernel size ' + str(N) + ' ' + str(src_type) + ' on ' + strPU)

dims = [N, N, N]
dimsDouble = [2*N, 2*N, 2*N]
dimsTuple = tuple(dims)
dimsDoubleTuple = tuple(dimsDouble)

#build test input in numpy (cupy does not have itemset)
src = np.ones(dimsTuple, dtype=src_type)
for k in range (np.size(src)):
    src.itemset(k, np.random.random()*10.0)

sym = np.zeros(dimsDoubleTuple, dtype=src_type)
for k in range (np.size(sym)):
    sym.itemset(k, np.random.random()*10.0)

xp = np
if forGPU:
    xp = cp
    #convert src and sym from NumPy to CuPy arrays
    src = cp.asarray(src)
    sym = cp.asarray(sym)

testSymCube = xp.fft.fftn(sym)

#original spinifel calculation
def orig_kernel_mdrfsconv(xp, src):
    ugrid_ups = xp.zeros((2*N,)*3, dtype=src.dtype)
    ugrid_ups[:N, :N, :N] = src
    F_ugrid_ups = xp.fft.fftn(xp.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * testSymCube
    ugrid_conv_out_ups = xp.fft.fftshift(xp.fft.ifftn(F_ugrid_conv_out_ups))
    return ugrid_conv_out_ups[:N, :N, :N]

fftx_result = fftx.convo.mdrfsconv(src, testSymCube)
spinifel_result = orig_kernel_mdrfsconv(xp, src)

max_spinifel = xp.max(xp.absolute(spinifel_result))
max_diff = xp.max(xp.absolute(spinifel_result - fftx_result))
print ('Relative diff between spinifel and FFTX kernels =  ' + str(max_diff/max_spinifel) )
