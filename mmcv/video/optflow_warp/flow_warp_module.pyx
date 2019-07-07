STUFF = "Hi"

import numpy as np
cimport numpy as np

np.import_array()

cdef extern from "flow_warp.hpp":
    void FlowWarp(double* img, double* flow1, double* out, const int height, const int width, const int channels, const int filling_value, const int interpolateMode)

def flow_warp_c(np.ndarray[double, ndim=3, mode="c"] img_array not None,
                np.ndarray[double, ndim=3, mode="c"] flow_array not None,
                int filling_value=0,
                int interpolate_mode=1):

    out_array = np.zeros_like(img_array)

    FlowWarp(<double*> np.PyArray_DATA(img_array),
             <double*> np.PyArray_DATA(flow_array),
             <double*> np.PyArray_DATA(out_array),
             out_array.shape[0],
             out_array.shape[1],
             out_array.shape[2],
             filling_value,
             interpolate_mode)

    return out_array
