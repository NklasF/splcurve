#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import _parametrization
from . import _knots
from . import _helpers
import numpy as np
import math


class Spline(object):
    """Definition of a B-Spline Curve in 2D.

    Parameters
    ----------
    t : array_like, (n+3+1,)
        Knots
    u : array_like, (n,)
        Parameters
    k : int
        Degree

    Attributes
    ----------
    t : ndarray
        Knot vector
    u : ndarray
        Parameter vector
    k : int
        Degree
    """

    def __init__(self, t, u, k):
        super(Spline, self).__init__()

        n = len(t) - self.k - 1

        self.t = np.asarray(t)
        self.u = np.asarray(u)
        self.k = k

        if (self.k < 0):
            raise ValueError('Negative degree is not possible')
        if n < self.k + 1:
            raise ValueError("Need at least %d knots for degree %d" %
                             (2*k + 2, k))
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if not np.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        # Not yet implemented
        # if self.c.ndim < 1:
        #     raise ValueError("Coefficients must be at least 1-dimensional.")
        # if self.c.shape[0] < n:
        #     raise ValueError("Knots, coefficients and degree are inconsistent.")

    @property
    def tuk(self):
        return self.t, self.u, self.k

    @classmethod
    def construct_fast(cls, t, u, k):
        """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        `t` and `u` must be of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.u, self.k = t, u, k
        return self

    def calc_bspl(self, x):
        """Compute BSplines at given point.

        Parameters
        ----------
        x : float
            The point at which the BSplines are to be evaluated

        Returns
        ----------
        bspl : ndarray, shape (k+1,)
            Non-zero BSplines at given point

        Notes
        -----
        BSpline are defined via:

        .. math::

            B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}
            B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                    + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

        References
        ----------
        .. [1] Carl de Boor, A practical guide to splines, Springer, 2001.
        """
        start = self.k
        end = len(self.t)-self.k-1
        if ((x < self.t[start]) or (x > self.t[end])):
            raise TypeError('Point x outside of the base interval')
        # Search on t[start:end-1] because of special case x == t[end]
        index = _helpers.binary_search(self.t, x, start, end-1)
        # Initialize bspl with k+1 entries for intermediate values
        bspl = np.zeros(self.k+1)

        # BSpline for degree 0
        bspl[0] = 1
        # Main Loop
        for j in range(self.k):
            # First entry has no north-west predecessor
            saved = 0
            # Loop for actual calculation of entries
            for r in range(j+1):
                deltar = self.t[index+r+1] - x
                deltal = x - self.t[index-j+r]
                term = bspl[r] / (deltar + deltal)
                # Internal neighbouring entries have common terms (saved)
                bspl[r] = saved + deltar * term
                saved = deltal * term
            # Last entry has no south-west predecessor
            bspl[j+1] = saved
        return bspl


def make_spline(points, p_type=0, k_type=0, k=3):
    """Compute a B-Spline Curve in 2D.

    Parameters
    ----------
    points : array_like (2, n)
        A list of sample vector arrays representing the curve in 2D
    p_type : int, optional
        See documentation of `_generate_param`
    k_type : int, optional
        See documentation of `_generate_knots`
    k : int, optional
        Degree of the BSpline curve

    Returns
    ----------
    spl :
        BSpline object.
    """
    if (len(points[0]) <= k) or (len(points[1]) <= k):
        raise ValueError('n > k must hold')
    if (len(points[0]) != len(points[1])):
        raise ValueError('Number of coordinates for points are unequal')
    if (k < 0):
        raise ValueError('Negative degree is not possible')
    x = np.asarray(points[0])
    y = np.asarray(points[1])

    u = _generate_param(x, y, p_type)
    t = _generate_knots(u, k, k_type)

    return Spline.construct_fast(t, u, k)


def _generate_param(x, y, p_type=0):
    """Compute the parameters of a B-Spline Curve in 2D.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissa
    y : array_like, shape (n,)
        Ordinate
    p_type : int, optional
        Determines the method for calculating the parametrization vector

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


def _generate_knots(u, k, k_type=0):
    """Compute the knots of a B-Spline Curve in 2D.

    Parameters
    ----------
    u : array_like, shape (n,)
        Parameters
    k : int
        Degree
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
    return _knots.calc_knots(u, k, k_type)
