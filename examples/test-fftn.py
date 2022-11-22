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
# 15 is OK, but 16 has FFTX being much slower.
itns = 20
ignored = 5

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("python test-fftn.py sz [ F|I [ itns [ignored [ d|s  [ GPU|CPU ]]]]]")
    print("  sz = N or N1,N2,N3")
    print("  F = Forward (default), I = Inverse")
    print("  itns = number of iterations, default " + str(itns))
    print("  ignored = number of initial iterations to ignore, default " + str(ignored))
    print("  d = double (default), s = single precision")
    sys.exit()

npar = 0

npar += 1
nnn = sys.argv[npar].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = (n1,n2,n3)

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

src = np.zeros(dims, dtype=np.complex128)
for k in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(k, vr + vi*1j)

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
    pyfunc = xp.fft.fftn
    fftxfunc = fftx.fft.fftn
    funcname = 'fftn'
else:
    pyfunc = xp.fft.ifftn
    fftxfunc = fftx.fft.ifftn
    funcname = 'ifftn'

    
print('**** Timing ' + funcname + ' on ' + dev + ', data type: ' + src.dtype.name + ', dims: ' + str(dims) + ' ****')
print('')
    

print(f'Timing {pymod} over {itns} itns, ignoring first {ignored}')
for i in range(itns):
    ts = time.perf_counter()
    resPy = pyfunc(src)
    tf = time.perf_counter()
    times_python[i] = tf - ts

# print(f'average {((tf - ts)/(itns*1.0)):0.6f}')
### print(np.array2string(times_python, separator=","))
print(f'average {np.average(times_python[ignored:itns])}')

# pytm = tf-ts
pytm = np.average(times_python[ignored:itns])
pytm_low = fftx.utils.avg_low(times_python, ignored, 2., 'times_python')
print(f'without outliers, average {pytm_low}')
print('')

print(f'Timing FFTX over {itns} itns, ignoring first {ignored}')
resC = None
for i in range(itns):
    ts = time.perf_counter()
    resC  = fftxfunc(src, resC)
    tf = time.perf_counter()
    times_fftx[i] = tf - ts

# print(f'average {((tf - ts)/(itns*1.0)):0.6f}')
### print(np.array2string(times_fftx, separator=","))
print(f'average {np.average(times_fftx[ignored:itns])}')

# fftxtm = tf-ts
fftxtm = np.average(times_fftx[ignored:itns])
fftxtm_low = fftx.utils.avg_low(times_fftx, ignored, 2., 'times_fftx')
print(f'without outliers, average {fftxtm_low}')
print('')

diffCP = xp.max( xp.absolute( resPy - resC ) )
maxCP = xp.max(xp.absolute(resPy))
print('Relative diff between ' + pymod + ' and FFTX transforms: ' + str(diffCP/maxCP))
print('Speedup (average after ignored) from ' + pymod + ' to FFTX: ' + f'{(pytm / fftxtm):0.2f}' + 'x')
print('Speedup (average without outliers) from ' + pymod + ' to FFTX: ' + f'{(pytm_low / fftxtm_low):0.2f}' + 'x')
print('')
