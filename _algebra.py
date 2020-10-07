#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#     Linear algebra helpers    #
#################################


def LU_solve(b, LU):
    """Finds a solution to the linear system Ax=b
    with A factorized into LU."""
    rhs = np.asarray(b)
    mat = np.asarray(LU)
    if (mat.shape[0] != mat.shape[1]):
        raise ValueError('Matrix must be square!')
    if (mat.shape[1] != rhs.shape[0]):
        raise ValueError('Matrix and Right-Side must be of same dimension!')
    size = rhs.shape

    # Solving Ly=rhs
    y = np.zeros(size)
    for i in range(size[0]):
        sum = 0
        for k in range(i):
            sum += mat[i, k]*y[k]
        y[i] = rhs[i] - sum

    # Solving Ux=y
    x = np.zeros(size)
    for i in range(size[0]-1, -1, -1):
        sum = 0
        for k in range(i+1, size[0]):
            sum += mat[i, k] * x[k]
        x[i] = (y[i] - sum) / mat[i, i]
    return x


def LU_fac(A):
    """Factorization of A according to A=LU without pivoting.
    L is an upper-triangular matrix.
    U is an lower-triangular matrix.
    LU=L+U-I"""
    mat = np.asarray(A)
    if (mat.shape[0] != mat.shape[1]):
        raise ValueError('Matrix must be square!')
    n = len(mat[0])
    LU = np.zeros((n, n))
    # Factorization of mat
    for i in range(n):
        # Calculate ith row
        for j in range(i, n):
            sum = 0.0
            for k in range(i):
                sum += LU[i, k]*LU[k, j]
            LU[i, j] = mat[i, j]-sum
        # Calculate ith column
        for j in range(i+1, n):
            sum = 0.0
            for k in range(i):
                sum += LU[j, k]*LU[k, i]
            LU[j, i] = (mat[j, i]-sum)/LU[i, i]
    return LU
