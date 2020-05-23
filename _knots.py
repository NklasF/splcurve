#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#  Interpolating spline helpers #
#################################


def knots_average(u):
    m = len(u) + 3 + 1
    if ((m - 8) > 0):
        v = np.zeros(m-8)
        for i in range(0, m-8):
            v[i] = (1/3) * sum([u[j] for j in range(i+1, i+4)])
        t = np.concatenate((4*[0], v, 4*[1]), axis=None)
        return t
    else:
        return np.concatenate((4*[0], 4*[1]), axis=None)


def knots_uniform(u):
    m = len(u) + 3 + 1
    if ((m - 8) > 0):
        v = np.zeros(m-8)
        for i in range(1, m-8+1):
            v[i-1] = i / (m-8+1)
        t = np.concatenate((4*[0], v, 4*[1]), axis=None)
        return t
    else:
        return np.concatenate((4*[0], 4*[1]), axis=None)


_switcher = {
    0: knots_average,
    1: knots_uniform
}


def calc_knots(u, k_type):
    # Get the function from switcher dictionary
    func = _switcher.get(k_type)
    if(not func):
        raise NameError('Invalid Type for Generation of Knots')
    # Execute the function
    return func(u)
