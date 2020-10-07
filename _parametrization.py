#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import _helpers
import numpy as np
import math

#################################
#  Interpolating spline helpers #
#################################


def param_uniform(x, y):
    n = len(x)
    v = np.zeros(n-2)
    for i in range(1, n-1):
        v[i-1] = i / (n-1)
    u = np.concatenate(([0], v, [1]), axis=None)
    return u


def param_foley(x, y):
    n = len(x)
    u = np.zeros(n)
    Q = _helpers.nielson_matrix([x, y])
    for i in range(0, n-1):
        left = 0
        right = 0
        d1 = _helpers.nielson_metric(Q, [x[i], y[i]], [x[i+1], y[i+1]])
        if (i != (n-2)):
            d2 = _helpers.nielson_metric(Q, [x[i+1], y[i+1]],
                                         [x[i+2], y[i+2]])
            d4 = _helpers.nielson_metric(Q, [x[i], y[i]],
                                         [x[i+2], y[i+2]])
            arc_in0 = (math.pow(d1, 2)+math.pow(d2, 2) -
                       math.pow(d4, 2))/(2*d2*d1)
            arc_in0 = arc_in0 if arc_in0 >= (-1) else -1
            th1 = math.pi - \
                np.arccos(arc_in0)
            theta1 = min(th1, math.pi/2)
            right = ((3*theta1*d2)/(2*(d1+d2)))
        if (i != 0):
            d0 = _helpers.nielson_metric(Q, [x[i-1], y[i-1]],
                                         [x[i], y[i]])
            d3 = _helpers.nielson_metric(Q, [x[i-1], y[i-1]],
                                         [x[i+1], y[i+1]])
            arc_in1 = (math.pow(d0, 2)+math.pow(d1, 2) -
                       math.pow(d3, 2))/(2*d1*d0)
            arc_in1 = arc_in1 if arc_in1 >= (-1) else -1
            th0 = math.pi - \
                np.arccos(arc_in1)
            theta0 = min(th0, math.pi/2)
            left = ((3*theta0*d0)/(2*(d0+d1)))
        u[i+1] = u[i] + d1 * \
            (1 + left + right)
    return u/u[n-1]


def param_chord(x, y):
    n = len(x)
    u = np.zeros(n)
    chords = 0
    for i in range(1, n):
        chords += math.sqrt((math.pow((x[i]-x[i-1]), 2)
                             + math.pow((y[i]-y[i-1]), 2)))
        u[i] = chords
    return u/u[n-1]


def param_centri(x, y, a=0.5):
    n = len(x)
    u = np.zeros(n)
    chords = 0
    for i in range(1, n):
        chords += math.pow(math.sqrt((math.pow((x[i]-x[i-1]), 2)
                                      + math.pow((y[i]-y[i-1]), 2))), a)
        u[i] = chords
    return u/u[n-1]


_switcher = {
    0: param_uniform,
    1: param_foley,
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
