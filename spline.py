#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import _parametrization
from . import _knots
from . import _helpers
from . import _algebra
import numpy as np
import math
import scipy.integrate
import scipy.optimize


class Spline(object):
    """Definition of a B-Spline curve in 2D.

    Parameters
    ----------
    t : array_like, (n+k+1, ...)
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

        self.t = np.asarray(t)
        self.c = np.asarray(c)
        self.k = k

        n = len(t) - self.k - 1

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

    def __call__(self, x, m=0):
        """Evaluate a spline function for up to the m-th derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the spline at
        m : int, optional
            Indicates the m-th derivative

        Returns
        ----------
        y : ndarray, shape (n, ...)
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        x = np.atleast_1d(x)
        y = np.empty((len(x), self.c.shape[1]), dtype=self.c.dtype)
        for i in range(len(x)):
            y[i] = self.de_Boor(x[i], m)
        return y

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
        """Compute B-Splines at given value for the m-th derivative.

        Parameters
        ----------
        x : float
            The value at which the BSplines are to be evaluated
        m : int, optional
            Indicates the m-th derivative

        Returns
        ----------
        bspl : ndarray, shape ((k-m)+1, ...)
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
        """Evaluate the coefficients of B-Splines at given value for up to the m-th derivative.

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
        # Special Case
        if (x == self.t[len(self.t)-self.k-1]):
            return points[self.k-m]
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
        """Evaluate the control points fo an interpolating spline curve given a set of data points and parameters.

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
        u : array_like (n, ...)
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


def haus_dist(points, spl, x):
    """Calculate the Hausdorff distance between a given B-Spline curve and a data polygon defined by a set of given points.

    Parameters
    ----------
    points : array_like (2, n)
        A list of sample vector arrays representing the data polygon in 2D
    spl :
        B-Spline object
    x : array_like
        Points to evaluate the Hausdorff distance at

    Returns
    ----------
    dist : float
        Hausdorff distance between the curve and the data polygon.
    """
    # Data polygon described by a B-Spline of degree 1
    data_poly = make_spline(points, k=1)

    # Points on the curve
    points1 = spl(x)
    # Points on the data polygon
    points2 = data_poly(x)

    n = len(points1)

    max12 = 0.0
    max21 = 0.0

    for i in range(n):
        min12 = float("inf")
        min21 = float("inf")
        for j in range(n):
            # Min Distance from the ith point on the curve (1) to the data polygon (2)
            dist12 = math.sqrt(
                (math.pow((points1[i][0]-points2[j][0]), 2) + math.pow((points1[i][1]-points2[j][1]), 2)))
            min12 = dist12 if dist12 < min12 else min12
            # Min Distance from the ith point on the data polygon (2) to the curve (1)
            dist21 = math.sqrt(
                (math.pow((points1[j][0]-points2[i][0]), 2) + math.pow((points1[j][1]-points2[i][1]), 2)))
            min21 = dist21 if dist21 < min21 else min21
        # Max of the Min Distances from curve (1) to data polygon (2)
        max12 = min12 if min12 > max12 else max12
        # Max of the Min Distances from data polygon (2) to curve (1)
        max21 = min21 if min21 > max21 else max21
    dist = max(max12, max21)
    return dist


def eval_angles(partitions, spl):
    """Calculate the approximated angles of the partitions of a given B-Spline curve. The partitions are defined by their chord lengths.

    Parameters
    ----------
    partitions : array_like (n, 2)
        A list of arrays with entries for angles and chord lengths representing the desired partitions of the curve
    spl :
        B-Spline object

    Returns
    ----------
    results : ndarray
        The approximated angles of the partitions of the curve.
    """
    parti = np.asarray(partitions)
    if ((len(parti.shape) != 2) or (parti.shape[1] != 2)):
        raise ValueError('Invalid information about partitions')
    results = np.zeros(parti.shape[0])
    # Integrand of the length of a B-spline curve

    def Len(x):
        return math.sqrt(math.pow(spl(x, m=1).T[0], 2)+math.pow(spl(x, m=1).T[1], 2))

    # Current chord
    chords = 0
    for i in range(len(parti)):
        # Skip linear partitions
        if (parti[i][0] == 0):
            results[i] = 0
        else:
            # Start and end value within the domain of the length
            start = chords
            end = (chords+parti[i][1])
            # Start and end value within the domain of the parameters
            u_start = scipy.optimize.brentq(
                lambda x: scipy.integrate.quad(Len, 0, x, limit=100)[0] - start, 0, 1)
            u_end = scipy.optimize.brentq(
                lambda x: scipy.integrate.quad(Len, 0, x, limit=100)[0] - end, 0, 1)
            # Derivative of the start point on the curve
            dX1, dY1 = spl(u_start, m=1).T
            # Derivative of the end point on the curve
            dX2, dY2 = spl(u_end, m=1).T
            # Intersection angle of the normal lines of the start and end point
            results[i] = np.degrees(np.arctan(
                np.absolute(((dX1/dY1)-(dX2/dY2))/(1+(dX1/dY1)*(dX2/dY2)))))
        # Update chord
        chords += parti[i][1]
    return results


def make_spline(points, p_type=0, k_type=0, k=3):
    """Compute a B-Spline curve in 2D.

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
        B-Spline object.
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
    """Compute the parameters of a B-Spline curve in 2D.

    Parameters
    ----------
    x : array_like, shape (n, ...)
        Abscissa
    y : array_like, shape (n, ...)
        Ordinate
    p_type : int, optional
        Determines the method for calculating the parametrization vector

        * if p_type=0, uniform spaced method
        * if p_type=1, foley-nielson method
        * if p_type=2, chord length method
        * if p_type=3, centripetal method

        The default value is 0.

    Returns
    ----------
    u : ndarray, shape (n, ...)
        Parametrization vector

    References
    ----------
    [1] Carl de Boor, A practical guide to splines, Springer, 2001.
    [2] Eugene TY Lee, Choosing nodes in parametric curve interpolation, Elsevier, 1989.
    [3] Thomas Foley and Gregory Nielson, Knot selection for parametric spline interpolation, Elsevier, 1989.
    """
    return _parametrization.calc_param(x, y, p_type)


def _generate_knots(u, k, k_type=0):
    """Compute the knots of a B-Spline curve in 2D.

    Parameters
    ----------
    u : array_like, shape (n, ...)
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
    t : ndarray, shape (n+3+1, ...)
        Knot vector
    """
    return _knots.calc_knots(u, k, k_type)
