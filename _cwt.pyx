'''
Created on 9 Oct 2013

@author: david
'''
#cython: cdivision=True

from __future__ import division

import numpy as np
from numpy import pi
cimport numpy as np

from libc.math cimport sqrt, exp, sin, cos, pow, M_PI

cimport cython

# Defining data types: --------
# time:
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# wavelet transform:
DTYPEC = np.complex128
ctypedef np.complex128_t DTYPEC_t

# Index
ctypedef Py_ssize_t index_t

# Input data. Could be float or double.
ctypedef fused data_t:
    np.float32_t                # FIXME: TypeError: No matching signature found
    np.float64_t                # if inputs have different dtype.
# ---------------------------

cdef inline DTYPEC_t morlet_i_c(double t, float W0):
    '''Complete inlined Morlet transform.'''
    return pow(pi, -0.25) * (morlet_i(t, W0) - exp(-0.5 * (t * t) - 0.5 * W0 * W0))

cdef inline DTYPEC_t morlet_i(double t, float W0):
    '''Incomplete inlined Morlet transform.'''
    return exp(-0.5 * (t * t)) * (cos(W0 * t) - 1j * sin(W0 * t))

cdef inline DTYPE_t ricker(double t):
    return 2 / (sqrt(3 * sqrt(M_PI))) * (1 - t * t) * exp(-0.5 * t * t)

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _mwt(np.ndarray[data_t, ndim=1] x, np.ndarray[data_t, ndim=1] scales):
    '''Perform Morlet continuous wavelet transform

    Input:
    -----
        x : 1-D np.ndarray, float32 or float64. The data to be transformed.
        scales: 1-D np.ndarray, float32 or float64. Scales at which to perform
             the transformation.

    Returns:
    -------
        out : the cwt of the x input.
    '''

    cdef:
        int N = scales.shape[0]
        int M = x.shape[0]
        double s, t_norm
        float W0 = 2.
        index_t i, tau, k
        DTYPEC_t integral_sum
        int kminus, k_plus

    # Defining the output file.
    cdef np.ndarray[DTYPEC_t, ndim = 2] out = np.empty((N, M), dtype=DTYPEC)

    for i in xrange(N):
        s = scales[i]
        for tau in xrange(M):
            integral_sum = 0

            # Only certain values of the integral contribute due to the
            # exponential decay. Allowing 7 sigma implies an error of 1e-13

            # We don't want to go outside of the limits:
            kminus = int_max(tau - 7 * int(s), 1)
            k_plus = int_min(tau + 7 * int(s), M - 1)

            for k in xrange(kminus, k_plus):
                t_norm = (k - tau) / s
                integral_sum += morlet_i_c(t_norm, W0) * x[k]

            # To make results coincide with trapezoidal rule, we also consider
            # the extrema, if necessary.
            if kminus == 1:
                integral_sum += 0.5 * morlet_i_c(-tau / s, W0) * x[0]

            if k_plus == M - 1:
                t_norm = (M - 1 - tau) / s
                integral_sum += 0.5 * morlet_i_c(t_norm, W0) * x[M - 1]

            # Normalise and store.
            out[i, tau] = integral_sum / sqrt(s)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _rwt(np.ndarray[data_t, ndim=1] x, np.ndarray[data_t, ndim=1] scales):
    '''Perform Ricker continuous wavelet transform

    Input:
    -----
        x : 1-D np.ndarray, float32 or float64. The data to be transformed.
        scales: 1-D np.ndarray, float32 or float64. Scales at which to perform
             the transformation.

    Returns:
    -------
        out : the cwt of the x input.
    '''

    cdef:
        int N = scales.shape[0]
        int M = x.shape[0]
        double s, t_norm
        index_t i, tau, k
        DTYPE_t integral_sum
        int kminus, k_plus
        int int_s

    # Defining the output file.
    cdef np.ndarray[DTYPE_t, ndim = 2] out = np.empty((N, M), dtype=DTYPE)

    for i in xrange(N):
        s = scales[i]
        for tau in xrange(M):
            integral_sum = 0.0

            # Only certain values of the integral contribute due to the
            # exponential decay. Allowing 8 sigma implies an error of 8e-14

            # We don't want to go outside of the limits:
            int_s = int(s)
            kminus = int_max(tau - 8 * int_s, 1)
            k_plus = int_min(tau + 8 * int_s, M - 1)

            for k in xrange(kminus, k_plus):
                t_norm = (k - tau) / s
                integral_sum += ricker(t_norm) * x[k]

            # To make results coincide with trapezoidal rule, we also consider
            # the extrema, if necessary.
            if kminus == 1:
                integral_sum += 0.5 * ricker(-tau / s) * x[0]

            if k_plus == M - 1:
                t_norm = (M - 1 - tau) / s
                integral_sum += 0.5 * ricker(t_norm) * x[M - 1]

            # Normalise and store.
            out[i, tau] = integral_sum / sqrt(s)

    return out