#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import _parametrization
from . import _knots
from . import _helpers
from . import _algebra
import numpy as np
import math as m


class Spline(object):
    """Definition of a B-Spline Curve in 2D.

    Parameters
    ----------
    t : array_like, (n+k+1,)
        Knots
    c : array_like, (n, ...)
        Coefficients
    k : int
        Degree

    Attributes
    ----------
    t : ndarray
        Knot vector
    c : ndarray
        Coefficients vector
    k : int
        Degree
    """

    def __init__(self, t, c, k):
        super(Spline, self).__init__()

        n = len(t) - self.k - 1

        self.t = np.asarray(t)
        self.c = np.asarray(c)
        self.k = k

        if (self.k < 0):
            raise ValueError('Negative degree is not possible')
        if (n < self.k + 1):
            raise ValueError("Need at least %d knots for degree %d" %
                             (2*k + 2, k))
        if (self.t.ndim != 1):
            raise ValueError("Knot vector must be one-dimensional.")
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if (not np.isfinite(self.t).all()):
            raise ValueError("Knots should not have nans or infs.")
        if (self.c.ndim < 1):
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if (self.c.shape[0] < n):
            raise ValueError(
                "Knots, coefficients and degree are inconsistent.")

    @property
    def tck(self):
        return self.t, self.c, self.k

    @classmethod
    def construct_fast(cls, t, c, k):
        """Construct a spline without making checks.
        Accepts same parameters as the regular constructor. Input arrays
        t and u must be of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.c, self.k = t, c, k
        return self

    def eval_bspl(self, x, m=0):
        """Compute BSplines at given value for the m-th derivative.

        Parameters
        ----------
        x : float
            The value at which the BSplines are to be evaluated
        m : int, optional
            Indicates the m-th derivative

        Returns
        ----------
        bspl : ndarray, shape ((k-m)+1,)
            BSplines at given value
        index : int
            index within the knot vector for the given value x

        References
        ----------
        [1] Carl de Boor, A practical guide to splines, Springer, 2001.
        """
        if (m > self.k):
            raise ValueError(
                'Negative degree after differentiation is not possible')
        # Search for knot span and return index
        index = self._search_index(x)
        # Initialize bspl for intermediate values
        bspl = np.zeros((self.k-m)+1)
        # BSpline for degree 0
        bspl[0] = 1
        # Main Loop
        for j in range(self.k-m):
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
        # bspl_i,k-m, for i=index, index-1, index-2, ... ,index-(k-m)
        return (bspl, index)

    def eval_coef(self, x, m=0):
        """Evaluate the coefficients of BSplines at given value for up to the m-th derivative.

        Parameters
        ----------
        x : float
            The value at which the coefficients of BSplines are to be evaluated
        m : int, optional
            Indicates the m-th derivative

        Returns
        ----------
        coef : ndarray, shape ((k-m)+1, ...)
            coefficients of BSplines at given value

        References
        ----------
        [1] Carl de Boor, A practical guide to splines, Springer, 2001.
        """
        if (m > self.k):
            raise ValueError(
                'Negative degree after differentiation is not possible')
        # Search for knot span and return index
        index = self._search_index(x)
        # Initialize coef to the coefficients of Bsplines
        coef = np.copy(self.c[index-self.k:index+1])
        # Main Loop
        for i in range(m):
            # Loop for actual calculation of coefficients
            for j in range(self.k-i):
                coef[j] = (self.k-i) * (coef[j+1] - coef[j]) / \
                    (self.t[j+index+1-i] - self.t[j+index-self.k+1])
        # coef^(m)_j, for j=index, index-1, index-2, ... ,index-(k-m)
        return coef[:self.k-m+1]

    def de_Boor(self, x, m=0):
        """Evaluate the spline function at given value for up to the m-th derivative.

        Parameters
        ----------
        x : float
            The value at which the coefficients of BSplines are to be evaluated
        m : int, optional
            Indicates the m-th derivative

        Returns
        ----------
        point : ndarray, shape (1, ...)
            Point on spline curve at given value

        References
        ----------
        [1] Carl de Boor, A practical guide to splines, Springer, 2001.
        """
        if (m > self.k):
            raise ValueError(
                'Negative degree after differentiation is not possible')
        # Search for knot span and return index
        index = self._search_index(x)
        # Initialize points to the coefficients of Bsplines according to m
        if (m == 0):
            points = np.copy(self.c[index-self.k:index+1])
        else:
            points = self.eval_coef(x, m)
        # Multiplicity of x, in case it is an internal knot
        s = np.count_nonzero(self.t == x)
        # Degree of spline curve after differentiation
        p = self.k - m
        # Main Loop
        for i in range(p - s):
            # Loop for actual calculation of points
            for j in range(p - s - i):
                # Left index for denominator of alpha
                left = j + index + 1
                # Right index for denominator of alpha
                right = j + index - p + i + 1
                alpha = (x - self.t[right]) / (self.t[left] - self.t[right])
                points[j] = (1 - alpha) * points[j] + alpha * points[j+1]
        return points[0]

    def interpolate(self, d, u):
        """Evaluate the control points fo an interpolating spline curve given a set of data points.

        Parameters
        ----------
        d : array_like, (n, ...)
            Data points
        u : array_like (n, )
            Parameter vector

        Returns
        ----------
        points : ndarray, shape (n, ...)
            Control points corresponding to the spline curve

        References
        ----------
        [1] Carl de Boor, A practical guide to splines, Springer, 2001.
        """
        param = np.asarray(u)
        # Data point D_i on row i
        dpoints = np.asarray(d)
        n = len(self.t) - self.k - 1
        if (param.ndim != 1):
            raise ValueError("Parameter vector must be one-dimensional.")
        if (len(param) != n):
            raise ValueError("Need exactly n parameters.")
        if (len(param) != dpoints.shape[0]):
            raise ValueError("Need exactly as many data points as parameters.")
        # Evaluate Collocation Matrix
        colloc = self._colloc(param)
        # Calculate Control Points
        cpoints = _algebra.LU_solve(dpoints, _algebra.LU_fac(colloc))
        return cpoints

    def _colloc(self, u):
        """Construct the collocation matrix according to the given parameter vector.

        Parameters
        ----------
        u : array_like (n, )
            Parameter vector

        Returns
        ----------
        colloc : ndarray, shape (n, n)
            Collocation Matrix
        """
        param = np.asarray(u)
        # Define shape of collocation matrix
        n = len(self.t) - self.k - 1
        if (param.ndim != 1):
            raise ValueError("Parameter vector must be one-dimensional.")
        if (len(param) != n):
            raise ValueError("Need at exactly n parameters.")
        colloc = np.zeros((n, n))
        for i in range(n):
            # Calculate B-splines
            bspl = self.eval_bspl(param[i])
            # Put B-splines into collocation matrix
            colloc[i][bspl[1]-self.k:bspl[1]+1] = bspl[0]
        return colloc

    def _search_index(self, x):
        """Find the index for a value within a knot span [t_index,t_index+1) of the knot vector.

        Parameters
        ----------
        x : float
            The value within a knot span on the base interval

        Returns
        ----------
        index : int
            index within the knot vector for the given value x
        """
        # Limits of the base interval
        start = self.k
        end = len(self.t)-self.k-1
        if ((x < self.t[start]) or (x > self.t[end])):
            raise ValueError('Value x outside of the base interval')
        # Search on t[start:end-1] because of special case x == t[end]
        return _helpers.binary_search(self.t, x, start, end-1)


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
    spline :
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
    spline = Spline.construct_fast(t, [], k)
    spline.c = spline.interpolate(np.array([x, y]).T, u)
    return spline


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
