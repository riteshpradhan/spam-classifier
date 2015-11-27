#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ritesh
# @Date:   2015-11-09 11:46:05
# @Last Modified by:   ritesh
# @Last Modified time: 2015-11-09 12:15:17

import numpy as np

def gini(expected, predicted):
    assert expected.shape[0] == predicted.shape[0], 'unequal number of rows'

    _all = np.asarray(np.c_[
        expected,
        predicted,
        np.arange(expected.shape[0])], dtype=np.float)

    _EXPECTED = 0
    _PREDICTED = 1
    _INDEX = 2

    # sort by predicted descending, then by index ascending
    sort_order = np.lexsort((_all[:, _INDEX], -1 * _all[:, _PREDICTED]))
    _all = _all[sort_order]

    total_losses = _all[:, _EXPECTED].sum()
    gini_sum = _all[:, _EXPECTED].cumsum().sum() / total_losses
    gini_sum -= (expected.shape[0] + 1.0) / 2.0
    return gini_sum / expected.shape[0]

def gini_normalized(expected, predicted):
    return gini(expected, predicted) / gini(expected, expected)

# a = np.array([5.1, 3.2, 1.7, 6.2, 8.1])
# b = np.array([3.1, 5.2, 2.7, 5.1, 1.1])

a = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
b = np.array([0, 1, 1, 0, 0, 0, 0, 1, 0, 1])


print('{:.6f}'.format(gini_normalized(a, b)))