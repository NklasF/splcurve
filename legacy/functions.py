from . import _parametrization
from . import _knots
from . import _helpers
import numpy as np
import math


def calc_bspl(self, x):
    """Compute non-zero BSplines at given point.

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
    if ((x < self.t[0]) or (x > self.t[len(self.t)-1])):
        raise TypeError('Extrapolation is not supported')
    index = _helpers.search_index(self.t, x)
    # Initialize bspl with k+1 entries for intermediate values
    bspl = np.zeros(self.k+1)

    bspl[0] = 1
    for j in range(self.k):
        saved = 0
        # Indicates to skip some iterations if necessary
        start = max(j - index, 0)
        stop = min(len(self.t) - index - 1, j+1)
        for r in range(start, stop):
            deltar = self.t[index+r+1] - x
            deltal = x - self.t[index-j+r]
            term = bspl[r] / (deltar + deltal)
            # Indicates an invalid Bspline, but prepares right term for next iteration
            if ((index - (j+1)) + r) == -1:
                bspl[r] = 0
            else:
                bspl[r] = saved + deltar * term
            saved = deltal * term
        # Indicates an invalid Bspline and sets it to 0
        if index > (len(self.t) - (j+1) - 2):
            bspl[j+1] = 0
        else:
            bspl[j+1] = saved
    # Slice bspl to contain only valid Bsplines
    beg = max(self.k - index, 0)
    end = min(len(self.t) - index - 1, self.k+1)
    return bspl[beg:end]
