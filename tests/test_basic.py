#!/usr/bin/env python
# -*- coding: utf-8 -*-

import splcurve.spline as spl
from splcurve import _algebra
from splcurve import _parametrization
from splcurve import _knots
import numpy as np
import math as m


def test_spl_curve():
    x = [0, 5, 3, 4, 5, 6, 7, 8]
    y = [0, 2, 4, 0, 4, 5, 2, 3]

    # points = np.array([m.sin(i / 3.0) for i in range(0, 11)])

    # knots = np.array([0, 0, 0, 0, 0.2, 0.3, 0.4, 0.5,
    #                   0.6, 0.7, 0.8, 1.0, 1.0, 1.0, 1.0])

    spl_curve = spl.make_spline([x, y], p_type=1, k_type=1, k=3)

    spl_curve.t = np.asarray([0, 0, 0, 0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])
    spl_curve.k = 3
    spl_curve.c = np.asarray([0, 2.0, -1.0, 4.0, 1.0, 0, 0])

    # Solution: f (0.4) = 1.813
    #           f'(0.4) = 11.2
    bspl = spl_curve.eval_bspl(0.4, 1)
    print("B-Spline:\t\t\t", bspl)

    coef = spl_curve.eval_coef(0.4, 1)
    print("Coefficients:\t\t", coef)

    print("Point on Curve (direct):\t\t", coef @ bspl[0])

    point1 = spl_curve.de_Boor([0.4, 0.4], 1)
    print("Point on Curve (deBoor):\t\t", point1)

    point2 = deBoorDerivative(4, 0.4, spl_curve.t, spl_curve.c, 3)
    print("Point on Curve (control):\t\t", point2)


def test_algebra():
    A = np.array([[2, 3, -1],
                  [4, 4, -3],
                  [-2, 3, -1]])
    # b = np.array([5, 3, 1])
    b = np.array([[5, 5],
                  [3, 3],
                  [1, 1]])

    # Solution: LU=[[ 2.  3. -1.]
    #               [ 2. -2. -1.]
    #               [-1. -3. -5.]]
    #           x=  [1. 2. 3.]

    LU = _algebra.LU_fac(A)
    print("LU Matrix:")
    print(LU)
    x = _algebra.LU_solve(b, LU)
    print("x Vector:")
    print(x)


def test_interpolation():
    x = [1, 2.5, 3, 4, 3, 6]
    y = [1, 3.5, 4.5, 3, 1.5, 0.5]
    dpoints = np.array([x, y]).T

    spl_curve = spl.make_spline([x, y], p_type=2, k_type=0, k=3)
    u = _parametrization.param_chord(x, y)
    colloc = spl_curve._colloc(u)
    print(colloc)
    cpoints = spl_curve.interpolate(dpoints, u)
    print(cpoints)
    print(spl_curve.c)


def test_parametrization():
    x = np.array([1, 2.5, 3, 4, 3, 6])
    y = np.array([1, 3.5, 4.5, 3, 1.5, 0.5])

    alpha = m.pi
    R = np.array([[m.cos(alpha), -m.sin(alpha)],
                  [m.sin(alpha), m.cos(alpha)]])
    points = R.dot(np.array([x, y]))

    print(np.array([x, y]))
    print(points)

    u1 = _parametrization.param_foley(x, y)
    print(u1)

    u2 = _parametrization.param_foley(points[0], points[1])
    print(u2)


def test_call():
    x = [1, 2.5, 3, 4, 3, 6]
    y = [1, 3.5, 4.5, 3, 1.5, 0.5]
    dpoints = np.array([x, y]).T

    spl_curve = spl.make_spline([x, y], p_type=2, k_type=0, k=3)
    u = _parametrization.param_chord(x, y)
    print(u)
    print(spl_curve.c)
    points = spl_curve(u)
    print(points)
