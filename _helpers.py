#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#        General helpers        #
#################################


def binary_search(t, x, start, end):
    mid = math.floor((start + end) / 2)
    if ((start - end) == 0) or ((t[mid] <= x) and (x < t[mid+1])):
        return mid
    elif t[mid] <= x:
        return binary_search(t, x, mid + 1, end)
    else:
        return binary_search(t, x, start, mid - 1)


def search_index(t, x):
    """Find the index `i` of `t` with :math:`t[i] <= x <= t[i+1]`.
    Under the assumption that `t` is nondecreasing :math:`t[j] <= t[j+1]`.
    """
    return binary_search(t, x, 0, len(t) - 1)
