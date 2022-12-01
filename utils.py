
import numpy as np
try:
    import cupy as cp
    cupy_found = True
except ModuleNotFoundError:
    cupy_found = False


def complexify1(array_in):
    # Call complexify(p, array_in) with p either CuPy or NumPy.
    if cupy_found and type(array_in) == cp.ndarray:
        return complexify(cp, array_in)
    else:
        return complexify(np, array_in)


def complexify(p, array_in):
    if array_in.dtype == p.complex128 and array_in.flags.c_contiguous:
        return array_in
    else:
        array_complex = p.ndarray(array_in.shape, p.complex128)
        array_complex[:, :, :] = array_in
        return array_complex

        
def max_abs_diff(p, array1, array2):
    result = p.max(p.absolute(array1 - array2))
    return result


def print_diff(p, array1, array2, textstr):
    maxdiff = max_abs_diff(p, array1, array2)
    if (maxdiff == 0.0):
        print(textstr + ' ' + str(array1.shape) + ' absolute diff 0.0')
    else:
        abs1 = p.max(p.absolute(array1))
        if (abs1 == 0.0):
            print(textstr + ' ' + str(array1.shape) + ' absolute-vs-0 diff ' +
                  str(p.max(p.absolute(array2))))
        else:
            reldiff = maxdiff/abs1
            if (reldiff == reldiff):
                # not NaN
                print(textstr + ' ' + str(array1.shape) + ' relative diff ' +
                      str(reldiff))
            else:
                # the NaN case
                print(textstr + ' ' + str(array1.shape) + ' max first ' +
                      str(p.max(p.absolute(array2))) + ' second ' +
                      str(abs1) + ' diff ' + str(maxdiff))
    return


def print_array_info(p, array_in, textstr):
    is_cupy = False
    if p.__name__ == "cupy":
        if isinstance(array_in, p._core.core.ndarray):
            is_cupy = True
    print(textstr +
          ' shape=' + str(array_in.shape) +
          ' dtype=' + str(array_in.dtype) +
          ' C=' + str(array_in.flags.c_contiguous) +
          ' CuPy=' + str(is_cupy))
    return


def avg_low(times, ignored, ratio, textstr):
    arrlen = np.size(times)
    tavg = np.average(times[ignored:arrlen])
    tot = 0
    count = 0
    for i in range(times.size):
        if (times[i] > ratio * tavg):
            if (len(textstr) > 0):
                print(f'{textstr}[{i}] = {times[i]}')
        elif (i >= ignored):
            tot += times[i]
            count += 1
    return (tot / count)


def symmetrize3d(p, src, dims):
    N0 = dims[0]
    N1 = dims[1]
    N2 = dims[2]
    sh = src.shape
    step0 = N0 if (p.mod(N0, 2) == 1) else N0 // 2
    hi0   = (N0 + 2) // 2
    lo0   = (N0 - 1) // 2 if (hi0 < sh[0]) else 0
    step1 = N1 if (p.mod(N1, 2) == 1) else N1 // 2
    hi1   = (N1 + 2) // 2
    lo1   = (N1 - 1) // 2 if (hi1 < sh[1]) else 0
    step2 = N2 if (p.mod(N2, 2) == 1) else N2 // 2
    hi2   = (N2 + 2) // 2
    lo2   = (N2 - 1) // 2 if (hi2 < sh[2]) else 0
    # 0-dimensional points
    src[::step0, ::step1, ::step2] = p.real(src[::step0, ::step1, ::step2])
    # 1-dimensional lines
    src[::step0, ::step1, hi2:] = p.conj(src[::step0, ::step1, lo2:0:-1]);
    src[::step0, hi1:, ::step2] = p.conj(src[::step0, lo1:0:-1, ::step2]);
    src[hi0:, ::step1, ::step2] = p.conj(src[lo0:0:-1, ::step1, ::step2]);
    # 2-dimensional planes
    src[::step0, 1:, hi2:] = p.conj(src[::step0, N1:0:-1, lo2:0:-1]);
    src[1:, ::step1, hi2:] = p.conj(src[N0:0:-1, ::step1, lo2:0:-1]);
    src[1:, hi1:, ::step2] = p.conj(src[N0:0:-1, lo1:0:-1, ::step2]);
    # 3-dimensional space
    src[1:, 1:, hi2:] =      p.conj(src[N0:0:-1, N1:0:-1, lo2:0:-1]);
    return
