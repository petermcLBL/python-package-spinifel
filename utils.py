def print_diff(p, array1, array2, textstr):
    maxdiff = p.max(p.absolute(array1 - array2))
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
