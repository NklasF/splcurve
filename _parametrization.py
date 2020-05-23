#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#  Interpolating spline helpers #
#################################


def param_identical(x, y):
    n = len(x)
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = v[i-1] + abs(x[i] - x[i-1])
    u = v / v[n-1]
    return u


def param_uniform(x, y):
    n = len(x)
    v = np.zeros(n-2)
    for i in range(1, n-1):
        v[i-1] = i / (n-1)
    u = np.concatenate(([0], v, [1]), axis=None)
    return u


def param_chord(x, y):
    n = len(x)
    L = sum([(math.sqrt(((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)))
             for i in range(1, n)])
    v = np.zeros(n-2)
    chords = 0
    for i in range(1, n-1):
        chords += math.sqrt(((x[i]-x[i-1])
                             ** 2 + (y[i]-y[i-1])**2))
        v[i-1] = chords / L
    u = np.concatenate(([0], v, [1]), axis=None)
    return u


def param_centri(x, y, a=0.5):
    n = len(x)
    L = sum([(math.pow((math.sqrt(((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2))), a))
             for i in range(1, n)])
    v = np.zeros(n-2)
    chords = 0
    for i in range(1, n-1):
        chords += math.pow(math.sqrt(((x[i]-x[i-1])
                                      ** 2 + (y[i]-y[i-1])**2)), a)
        v[i-1] = chords / L
    u = np.concatenate(([0], v, [1]), axis=None)
    return u


_switcher = {
    0: param_identical,
    1: param_uniform,
    2: param_chord,
    3: param_centri
}


def calc_param(x, y, p_type):
    # Get the function from switcher dictionary
    func = _switcher.get(p_type)
    if(not func):
        raise NameError('Invalid Type for Parametrization')
    # Execute the function
    return func(x, y)
