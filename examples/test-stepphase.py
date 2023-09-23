#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time
import fftx
import sys

import snowwhite as sw
from snowwhite.stepphasesolver import *

src_type = np.double
c_type = 'double'

itns = 20
ignored = 5

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("python test-stepphase.py sz [ itns [ignored [ d|s  [ GPU|CPU ]]]]")
    print("  sz = N")
    print("  itns = number of iterations, default " + str(itns))
    print("  ignored = number of initial iterations to ignore, default " + str(ignored))
    print("  d = double (default), s = single precision")
    sys.exit()

npar = 0

npar += 1
if len(sys.argv) > npar:
    N = int(sys.argv[npar])

dims = [N, N, N]
dimsTuple = tuple(dims)

npar += 1
if len(sys.argv) > npar:
    itns = int(sys.argv[npar])

npar += 1
if len(sys.argv) > npar:
    ignored = int(sys.argv[npar])

npar += 1
if len(sys.argv) > npar:
    if sys.argv[npar] == "s":
        src_type = np.single
        c_type = 'float'

npar += 1
if len ( sys.argv ) > npar:
    plat_arg = sys.argv[npar]
else:
    plat_arg = "GPU" if (cp != None) else "CPU"
    
if plat_arg == "GPU" and (cp != None):
    forGPU = True
    xp = cp
else:
    forGPU = False 
    xp = np       

#original spinifel calculation
def orig_kernel_stepphase(xp, src):
    amps_full = xp.absolute(xp.fft.fftn(src))**3
    rho_hat = xp.fft.fftn(src)
    phases = xp.angle(rho_hat)
    amp_mask = xp.ones(dims, dtype=xp.bool_)
    amp_mask[0, 0, 0] = 0
    rho_hat_mod = xp.where(amp_mask, amps_full*xp.exp(1j*phases), rho_hat)
    return xp.fft.ifftn(rho_hat_mod).real

#build test input in numpy (cupy does not have itemset)
src = np.ones(dimsTuple, dtype=src_type)
for  k in range (np.size(src)):
    src.itemset(k, np.random.random()*10.0)

if forGPU:
    xp = cp
    src = cp.asarray(src)
    dev = 'GPU'
    pymod = 'CuPy'
else:
    xp = np
    dev = 'CPU'
    pymod = 'NumPy'

#set amplitudes to a function of |fft(src)|
amplitudes = xp.absolute(xp.fft.rfftn(src))**3
amps_full = xp.absolute(xp.fft.fftn(src))**3

times_spinifel = np.zeros(itns)
times_fftx = np.zeros(itns)

if forGPU:
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

print('**** Timing phasing kernels on ' + dev + ', data type: ' + src.dtype.name + ', dims: ' + str(dims) + ' ****')
print('')
    
platform = SW_HIP if sw.has_ROCm() else SW_CUDA
opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : platform }

p1 = StepPhaseProblem(N)
s1 = StepPhaseSolver(p1, opts)

func = lambda s, a, d : s1.solve(s, a, d)

fftx_result = None
for i in range(itns):
    # original spinifel calculation
    ts = time.perf_counter()
    if forGPU:
        start_gpu.record()
    spinifel_result = orig_kernel_stepphase(xp, src)
    if forGPU:
        end_gpu.record()
        end_gpu.synchronize()
        # cp.cuda.get_elapsed_time returns time in millisec; convert to sec.
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) * 0.001
    tf = time.perf_counter()
    times_spinifel[i] = t_gpu if (forGPU) else tf - ts
    # FFTX
    ts = time.perf_counter()
    if forGPU:
        start_gpu.record()
    fftx_result = func(src, amplitudes, fftx_result)
    if forGPU:
        end_gpu.record()
        end_gpu.synchronize()
        # cp.cuda.get_elapsed_time returns time in millisec; convert to sec.
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) * 0.001
    tf = time.perf_counter()
    times_fftx[i] = t_gpu if (forGPU) else tf - ts

print(f'Timing kernels over {itns} itns, ignoring first {ignored}, in seconds')
tavg_spinifel = np.average(times_spinifel[ignored:itns])
print(f'SpiniFEL average {tavg_spinifel}')
tavg_fftx = np.average(times_fftx[ignored:itns])
print(f'FFTX average {tavg_fftx}')
print('')

max_spinifel = xp.max(xp.absolute(spinifel_result))
max_diff = xp.max( xp.absolute( spinifel_result - fftx_result ) )
print ('Relative diff between spinifel and FFTX kernels: ' +
       str(max_diff/max_spinifel) )
print('Speedup (average after ignored) from Spinifel to FFTX: ' +
      f'{(tavg_spinifel / tavg_fftx):0.2f}' + 'x')
print('')
