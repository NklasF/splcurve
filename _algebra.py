#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#     Linear algebra helpers    #
#################################


def back_sub(b, A):
    """Finds a solution to the linear system Ax=b,
    if A is an upper triangular matrix.
    """
    n = len(b)-1
    x = np.zeros(n+1)
    x[n] = b[n] / A[n, n]
    for k in range(n-1, -1, -1):
        s = 0
        for j in range(k+1, n+1):
            s += (A[k, j] * x[j])
        x[k] = (b[k] - s) / A[k, k]
    return x
