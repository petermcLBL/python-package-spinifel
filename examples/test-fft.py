#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time
import fftx
import sys

FORWARD = True
cxtype = np.cdouble
itns = 20
ignored = 5

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("python test-fft.py sz [ F|I [ itns [ignored [ d|s  [ GPU|CPU ]]]]]")
    print("  sz = length of FFT")
    print("  F = Forward (default), I = Inverse")
    print("  itns = number of iterations, default " + str(itns))
    print("  ignored = number of initial iterations to ignore, default " + str(ignored))
    print("  d = double (default), s = single precision")
    sys.exit()

npar = 0

npar += 1
n = int(sys.argv[npar])

npar += 1
if len(sys.argv) > npar:
    if sys.argv[npar] == "I":
        FORWARD = False
        
npar += 1
if len(sys.argv) > npar:
    itns = int(sys.argv[npar])

npar += 1
if len(sys.argv) > npar:
    ignored = int(sys.argv[npar])

npar += 1
if len(sys.argv) > npar:
    if sys.argv[npar] == "s":
        cxtype = np.csingle

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

times_python = np.zeros(itns)
times_fftx = np.zeros(itns)

if forGPU:
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()


src = np.zeros(n, cxtype)
for k in range (n):
    vr = np.random.random()
    vi = np.random.random()
    src[k] = vr + vi*1j

if forGPU:
    xp = cp
    src = cp.asarray(src)
    dev = 'GPU'
    pymod = 'CuPy'
else:
    xp = np
    dev = 'CPU'
    pymod = 'NumPy'

if FORWARD:
    pyfunc = xp.fft.fft
    fftxfunc = fftx.fft.fft
    funcname = 'fft'
else:
    pyfunc = xp.fft.ifft
    fftxfunc = fftx.fft.ifft
    funcname = 'ifft'

    
print('**** Timing ' + funcname + ' on ' + dev + ', data type: ' + src.dtype.name + ', dim: ' + str(n) + ' ****')
print('')
    

got_Py = True
print(f'Timing {pymod} over {itns} itns, ignoring first {ignored}')
for i in range(itns):
    ts = time.perf_counter()
    if forGPU:
        start_gpu.record()
    try:
        resPy = pyfunc(src)
    except RuntimeError:
        got_Py = False
        break
    if forGPU:
        end_gpu.record()
        end_gpu.synchronize()
        # cuda.get_elapsed_time returns time in millisec; convert to sec.
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) * 0.001
    tf = time.perf_counter()
    times_python[i] = t_gpu if (forGPU) else tf - ts

# print(f'average {((tf - ts)/(itns*1.0)):0.6f}')
### print(np.array2string(times_python, separator=","))
if got_Py:
    print(f'average {np.average(times_python[ignored:itns])}')
    # pytm = tf-ts
    pytm = np.average(times_python[ignored:itns])
    pytm_low = fftx.utils.avg_low(times_python, ignored, 2., 'times_python')
    print(f'without outliers, average {pytm_low}')
else:
    print('average NaN')
    print('without outliers, average NaN')
print('')

print(f'Timing FFTX over {itns} itns, ignoring first {ignored}')
got_FFTX = True
resC = None
for i in range(itns):
    ts = time.perf_counter()
    if forGPU:
        start_gpu.record()
    try:
        resC  = fftxfunc(src, resC)
    except RuntimeError:
        got_FFTX = False
        break
    if forGPU:
        end_gpu.record()
        end_gpu.synchronize()
        # cp.cuda.get_elapsed_time returns time in millisec; convert to sec.
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) * 0.001
    tf = time.perf_counter()
    times_fftx[i] = t_gpu if (forGPU) else tf - ts

# print(f'average {((tf - ts)/(itns*1.0)):0.6f}')
### print(np.array2string(times_fftx, separator=","))
if got_FFTX:
    print(f'average {np.average(times_fftx[ignored:itns])}')
    # fftxtm = tf-ts
    fftxtm = np.average(times_fftx[ignored:itns])
    fftxtm_low = fftx.utils.avg_low(times_fftx, ignored, 2., 'times_fftx')
    print(f'without outliers, average {fftxtm_low}')
else:
    print('average NaN')
    print('without outliers, average NaN')
print('')

if (got_FFTX & got_Py):
    diffCP = xp.max( xp.absolute( resPy - resC ) )
    maxCP = xp.max(xp.absolute(resPy))
    print('Relative diff between ' + pymod + ' and FFTX transforms: ' +
          str(diffCP/maxCP))
    print('Speedup (average after ignored) from ' + pymod + ' to FFTX: ' +
          f'{(pytm / fftxtm):0.2f}' + 'x')
    print('Speedup (average without outliers) from ' + pymod + ' to FFTX: ' +
          f'{(pytm_low / fftxtm_low):0.2f}' + 'x')
else:
    # These lines make it easier for a script to parse output.
    status_Py = pymod + (" succeeded" if (got_Py) else " failed")
    status_FFTX = "FFTX" + (" succeeded" if (got_FFTX) else " failed")
    print('Relative diff ' + status_Py + ' and ' + status_FFTX + ': NaN')
    print('Speedup (average after ignored) from ' + pymod + ' to FFTX: NaNx')
    print('Speedup (average without outliers) from ' + pymod + ' to FFTX: NaNx')
print('')
