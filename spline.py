#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import _parametrization
from . import _knots
import numpy as np
import math


class Spline(object):
    """Definition of a cubic B-Spline Curve in 2D.

    Parameters
    ----------
    t : array_like, (n+3+1,)
        Knots
    u : array_like, (n,)
        Parameters

    Attributes
    ----------
    t : ndarray
        Knot vector
    u : ndarray
        Parameter vector
    """

    def __init__(self, t, u):
        super(Spline, self).__init__()

        self.t = np.asarray(t)
        self.u = np.asarray(u)

    @property
    def tu(self):
        return self.t, self.u

    @classmethod
    def construct_fast(cls, t, u):
        """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        `t` and `u` must be of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.u = t, u
        return self


def make_spline(points, p_type=0, k_type=0):
    """Compute a cubic B-Spline Curve in 2D.

    Parameters
    ----------
    points : array_like (2, n)
        A list of sample vector arrays representing the curve in 2D.
    p_type : int, optional
        See documentation of `_generate_param`
    k_type : int, optional
        See documentation of `_generate_knots`

    Returns
    ----------
    spl :
        a cubic BSpline object.
    """
    if (len(points[0]) <= 3) or (len(points[1]) <= 3):
        raise TypeError('n > 3 must hold')
    if (len(points[0]) != len(points[1])):
        raise TypeError('Number of coordinates for points are unequal')
    x = np.asarray(points[0])
    y = np.asarray(points[1])

    u = _generate_param(x, y, p_type)
    t = _generate_knots(u, k_type)

    return Spline.construct_fast(t, u)


def _generate_param(x, y, p_type=0):
    """Compute the parameters of a cubic B-Spline Curve in 2D.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissa
    y : array_like, shape (n,)
        Ordinate
    p_type : int, optional
        Determines the method for calculating the parametrization vector.

        * if p_type=0, identically spaced method
        * if p_type=1, uniform spaced method
        * if p_type=2, chord length method
        * if p_type=3, centripetal method

        The default value is 0.

    Returns
    ----------
    u : ndarray, shape (n,)
        Parametrization vector
    """
    return _parametrization.calc_param(x, y, p_type)


def _generate_knots(u, k_type=0):
    """Compute the knots of a cubic B-Spline Curve in 2D.

    Parameters
    ----------
    u : array_like, shape (n,)
        Parameters
    u_type : int, optional
        Determines the method for calculating the parametrization vector.

        * if u_type=0, averaging method
        * if u_type=1, uniform spaced method

        The default value is 0.

    Returns
    ----------
    t : ndarray, shape (n+3+1,)
        Knot vector
    """
    return _knots.calc_knots(u, k_type)
