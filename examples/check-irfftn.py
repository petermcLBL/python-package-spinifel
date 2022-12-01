#! python

import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None
import time
import fftx
import sys

ftype = np.double
cxtype = np.cdouble

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    print("python check-irfftn.py sz [ d|s [ GPU|CPU ]]")
    print("  sz = N or N1,N2,N3")
    print("  d = double (default), s = single precision")
    sys.exit()

npar = 0

npar += 1
nnn = sys.argv[npar].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = [n1,n2,n3]

npar += 1
if len(sys.argv) > npar:
    if sys.argv[npar] == "s":
        ftype = np.single
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
        
dims2 = dims.copy()
z = dims2.pop()
dims2.append(z // 2 + 1)

print(f'dims = {dims}')
print(f'dims2 = {dims2}')

### Try on truncated array of dimensions dims2

src_trunc = xp.ones(tuple(dims2), cxtype)
for k in range (xp.size(src_trunc)):
    vr = xp.random.random()
    vi = xp.random.random()
    src_trunc.itemset(k,vr + vi * 1j)

fftx.utils.symmetrize3d(xp, src_trunc, dims)
dst_trunc = xp.fft.irfftn(src_trunc, dims)
# Now apply real-to-complex FFT on dst_trunc, to get a truncated array
# that should be the same as the one we started with.
src_trunc_back = xp.fft.rfftn(dst_trunc)

src_trunc_max = xp.max(xp.absolute(src_trunc))
diff_trunc_max = xp.max(xp.absolute(src_trunc_back - src_trunc))
reldiff_trunc = diff_trunc_max / src_trunc_max
print(f'For truncated array {src_trunc.shape}, relative diff = {reldiff_trunc}')

### Try on full array of dimensions dims

src_full = xp.ones(tuple(dims), cxtype)
for k in range (xp.size(src_full)):
    vr = xp.random.random()
    vi = xp.random.random()
    src_full.itemset(k,vr + vi * 1j)

fftx.utils.symmetrize3d(xp, src_full, dims)
dst_full = xp.fft.irfftn(src_full, dims)

# Now apply real-to-complex FFT on dst_full, to get a full array
# that should be the same as the one we started with.
src_full_back = xp.fft.fftn(dst_full)

src_full_max = xp.max(xp.absolute(src_full))
diff_full_max = xp.max(xp.absolute(src_full_back - src_full))
reldiff_full = diff_full_max / src_full_max
print(f'For full array {src_full.shape}, relative diff = {reldiff_full}')
