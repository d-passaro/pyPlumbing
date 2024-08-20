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

# The rest of your function follows here
def L_norm_cython_slow(np.ndarray[np.float64_t, ndim=2] inv_plumbing, 
                      np.ndarray[np.float64_t, ndim=2] C_inv, 
                      np.ndarray[np.float64_t, ndim=3] ell_array, 
                      int rnd):
    """
    Cython version of the vectorized L norm of a plumbing graph, 
    with optimized dot product computation and cached inverse.
    """
    # Ensure ell_array is a NumPy array with dtype float64
    ell_array = np.asarray(ell_array, dtype=np.float64)

    # Precompute the dot products using einsum
    cdef np.ndarray[np.float64_t, ndim=3] ell_C_inv = np.einsum("ij,klj->kli", C_inv, ell_array)

    # Perform matrix multiplication
    cdef np.ndarray[np.float64_t, ndim=3] product = np.matmul(ell_C_inv, np.transpose(ell_array, (0, 2, 1)))

    # Compute the total sum
    cdef np.ndarray[np.float64_t, ndim=1] tot = np.sum(inv_plumbing * product, axis=(1, 2))

    # Round the result to the specified number of decimals
    cdef np.ndarray[np.float64_t, ndim=1] rounded_tot = np.round(tot, decimals=rnd)

    return rounded_tot


@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative index wrapping
def L_norm_cython_array(np.ndarray[np.float64_t, ndim=2] inv_plumbing, 
                  np.ndarray[np.float64_t, ndim=2] C_inv, 
                  np.ndarray[np.float64_t, ndim=3] ell_array, 
                  int rnd):
    """
    Optimized Cython version of the vectorized L norm of a plumbing graph, 
    with reduced overhead and potential parallelism.
    """

    cdef int n_plumbing = inv_plumbing.shape[0]
    cdef int n_ell = ell_array.shape[0]
    cdef int rows_ell = ell_array.shape[1]
    cdef int cols_ell = ell_array.shape[2]
    cdef int rows_C_inv = C_inv.shape[1]

    # Declare variables for the loops
    cdef int i, j, k, l

    # Preallocate arrays for the computations
    cdef np.ndarray[np.float64_t, ndim=3] ell_C_inv = np.zeros((n_ell, rows_ell, cols_ell), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=3] product = np.zeros((n_ell, rows_ell, rows_ell), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] tot = np.zeros(n_ell, dtype=np.float64)

    # Compute ell_C_inv (equivalent to einsum) - Avoid einsum for better performance
    with nogil:
        for k in prange(n_ell):
            for l in range(rows_ell):
                for i in range(cols_ell):
                    ell_C_inv[k, l, i] = 0.0
                    for j in range(rows_C_inv):
                        ell_C_inv[k, l, i] += C_inv[j, i] * ell_array[k, l, j]

    # Compute product (equivalent to np.matmul) inside the GIL since it involves array slicing
    with nogil:
        for k in prange(n_ell):
            for l in range(rows_ell):
                for i in range(rows_ell):  # Matching the first dimension of ell_array for transpose
                    product[k, l, i] = 0.0
                    for j in range(cols_ell):
                        product[k, l, i] += ell_C_inv[k, l, j] * ell_array[k, i, j]

    # Perform the summation and multiply by inv_plumbing
    with nogil:
        for k in prange(n_ell):
            tot[k] = 0.0
            for i in range(n_plumbing):
                for j in range(n_plumbing):
                    tot[k] += inv_plumbing[i, j] * product[k, i, j]

    # The GIL is already held here, no need to explicitly acquire it
    return np.round(tot, decimals=rnd)


@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative index wrapping
def L_norm_cython_array_combined(np.ndarray[np.float64_t, ndim=2] inv_plumbing, 
                  np.ndarray[np.float64_t, ndim=2] C_inv, 
                  np.ndarray[np.float64_t, ndim=3] ell_array, 
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

    # Preallocate arrays for the computations
    cdef np.ndarray[np.float64_t, ndim=1] tot = np.zeros(n_ell, dtype=np.float64)

    with nogil:
        for k in prange(n_ell):
            tot[k] = 0.0
            for l in range(n_vertices):
                for i in range(n_vertices):
                    for s in range(rk):
                        for j in range(rk):
                            tot[k] += ell_array[k,l,s] * C_inv[s, j] * ell_array[k,i,j] * inv_plumbing[l,i]

    # The GIL is already held here, no need to explicitly acquire it
    return np.round(tot, decimals=rnd)


# Define a GIL-free rounding function
cdef double c_round(double x, int decimals) nogil:
    cdef double factor = 10.0 ** decimals
    return floor(x * factor + 0.5) / factor

@cython.boundscheck(False)  # Disable bounds-checking for performance
@cython.wraparound(False)   # Disable negative index wrapping
def L_norm_cython_combined(double[:,:] inv_plumbing, 
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
