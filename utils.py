
import numpy as np

def make_numpy(p, array_in):
    if p.__name__ == "cupy":
        if isinstance(array_in, p._core.core.ndarray):
            return p.asnumpy(array_in)
        else:
            return array_in
    else:
        return array_in


def complex_numpy(p, array_in):
    array_numpy = make_numpy(p, array_in)
    #if array_numpy.dtype == np.complex128:
    #    return array_numpy
    #else:
    array_complex = np.ndarray(array_numpy.shape, np.complex128)
    array_complex[:, :, :] = array_numpy
    return array_complex

        
def max_abs_diff(p, array1, array2):
    array1_numpy = make_numpy(p, array1)
    array2_numpy = make_numpy(p, array2)
    # np here because these are NumPy arrays
    result = np.max(np.absolute(array1_numpy - array2_numpy))
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
