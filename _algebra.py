#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#     Linear algebra helpers    #
#################################


def LU_solve(b, LU):
    """Finds a solution to the linear system Ax=b
    with A factorized into LU.
    """
    if (len(LU[0]) != len(LU[1])):
        raise ValueError('Matrix must be square!')
    if (len(LU[0]) != len(b)):
        raise ValueError('Matrix and Right-Side must be of same dimension!')
    n = len(b)

    # Solving Ly=b
    y = np.zeros(n)
    for i in range(n):
        sum = 0
        for k in range(i):
            sum += LU[i, k]*y[k]
        y[i] = b[i] - sum

    # Solving Ux=y
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        sum = 0
        for k in range(i+1, n):
            sum += LU[i, k] * x[k]
        x[i] = (y[i] - sum) / LU[i, i]
    return x


def LU_fac(A):
    """Factorization of A according to A=LU without pivoting.
    L is an upper-triangular matrix.
    U is an lower-triangular matrix.
    LU=L+U-I
    """
    if (len(A[0]) != len(A[1])):
        raise ValueError('Matrix must be square!')
    n = len(A[0])
    LU = np.zeros((n, n))
    # Factorization of A
    for i in range(n):
        # Calculate ith row
        for j in range(i, n):
            sum = 0.0
            for k in range(i):
                sum += LU[i, k]*LU[k, j]
            LU[i, j] = A[i, j]-sum
        # Calculate ith column
        for j in range(i+1, n):
            sum = 0.0
            for k in range(i):
                sum += LU[j, k]*LU[k, i]
            LU[j, i] = (A[j, i]-sum)/LU[i, i]
    return LU
