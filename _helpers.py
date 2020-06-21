#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

#################################
#        General helpers        #
#################################


def search_index(t, x):
    """Find the index `i` of `t` with :math:`t[i] \le x < t[i+1]`.
    Under the assumption that `t` is nondecreasing :math:`t[i] < t[i+1]`.
    """
    size = len(t)
    for i in range(size):
        if x >= t[size-i-1]:
            return size-i-1
