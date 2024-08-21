# cython_numpy_example.pyx

# Disable deprecated NumPy API
# You must place this before importing numpy's C API
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.math cimport floor

# Define a GIL-free rounding function
cdef double c_round(double x, int decimals) nogil:
    cdef double factor = 10.0 ** decimals
    return floor(x * factor + 0.5) / factor

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative index wrapping
def L_norm_cython(double[:,:] inv_plumbing, 
                  double[:,:] C_inv, 
                  double[:,:,:] ell_array, 
                  int rnd):
    """
    Optimized Cython version of the vectorized L norm of a plumbing graph, 
    with reduced overhead and potential parallelism.
    """

    cdef int n_ell = ell_array.shape[0]
    cdef int n_vertices = ell_array.shape[1]
    cdef int rk = ell_array.shape[2]

    # Declare variables for the loops
    cdef int k, l, i, s, j

    # Declare a memoryview for tot
    cdef double[:] tot = np.zeros(n_ell)

    with nogil:
        for k in prange(n_ell):
            tot[k] = 0.0
            for l in range(n_vertices):
                for i in range(n_vertices):
                    for s in range(rk):
                        for j in range(rk):
                            tot[k] += ell_array[k,l,s] * C_inv[s, j] * ell_array[k,i,j] * inv_plumbing[l,i]
            tot[k] = c_round(tot[k], rnd)
    return tot
