
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
