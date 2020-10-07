#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#        General helpers        #
#################################


def binary_search(t, x, start, end):
    """Find the index i of t with t[i] <= x <= t[i+1].
    Under the assumption that t is nondecreasing t[j] <= t[j+1]."""
    mid = math.floor((start + end) / 2)
    if ((start - end) == 0) or ((t[mid] <= x) and (x < t[mid+1])):
        return mid
    elif t[mid] <= x:
        return binary_search(t, x, mid + 1, end)
    else:
        return binary_search(t, x, start, mid - 1)


def nielson_metric(Q, p1, p2):
    U = np.asarray(p1)
    V = np.asarray(p2)
    return math.sqrt((U-V).dot(Q.dot(U-V)))


def nielson_matrix(points):
    x = np.asarray(points[0])
    y = np.asarray(points[1])
    n = len(x)
    x_hat = np.sum(x) / n
    y_hat = np.sum(y) / n
    v_x = sum([(x[i]-x_hat)**2 for i in range(0, n)]) / n
    v_y = sum([(y[i]-y_hat)**2 for i in range(0, n)]) / n
    v_xy = sum([(x[i]-x_hat)*(y[i]-y_hat) for i in range(0, n)]) / n
    g = v_x * v_y - v_xy**2
    Q = np.array([[v_y/g, -(v_xy/g)],
                  [-(v_xy/g), v_x/g]])
    return Q
