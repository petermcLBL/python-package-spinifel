
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
        print(textstr + ' ' + str(array1.shape) + ' relative diff ' +
              str(maxdiff/abs1))
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
