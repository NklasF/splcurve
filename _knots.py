#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#  Interpolating spline helpers #
#################################


def knots_average(u, k):
    m = len(u) + k + 1
    if ((m - (2*(k+1))) > 0):
        v = np.zeros(m - (2*(k+1)))
        for i in range(0, m - (2*(k+1))):
            v[i] = (1/k) * sum([u[j] for j in range(i+1, i+k+1)])
        t = np.concatenate(((k+1)*[0], v, (k+1)*[1]), axis=None)
        return t
    else:
        return np.concatenate(((k+1)*[0], (k+1)*[1]), axis=None)


def knots_uniform(u, k):
    m = len(u) + k + 1
    if ((m - (2*(k+1))) > 0):
        v = np.zeros(m - (2*(k+1)))
        for i in range(1, m - (2*(k+1))+1):
            v[i-1] = i / (m - (2*(k+1))+1)
        t = np.concatenate(((k+1)*[0], v, (k+1)*[1]), axis=None)
        return t
    else:
        return np.concatenate(((k+1)*[0], (k+1)*[1]), axis=None)


_switcher = {
    0: knots_average,
    1: knots_uniform
}


def calc_knots(u, k, k_type):
    # Get the function from switcher dictionary
    func = _switcher.get(k_type)
    if(not func):
        raise NameError('Invalid Type for Generation of Knots')
    # Execute the function
    return func(u, k)
