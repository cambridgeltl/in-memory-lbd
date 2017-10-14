from __future__ import print_function
from __future__ import absolute_import

cimport cython

from cpython cimport array

import array

from itertools import izip


# Temporarily use ints for scores
ctypedef int score_t


def open_discovery_core(int a_idx,
                        list neighbour_idx,
                        list weights_from,
                        array.array score,
                        array.array is_c_idx,
                        array.array exclude_idx,
                        filter_node,
                        agg, acc):
    cdef int b_idx, c_idx
    cdef score_t e1_weight, e2_weight

    cdef float[:] c_score = score
    cdef signed char[:] c_is_c_idx = is_c_idx
    cdef signed char[:] c_exclude_idx = exclude_idx
    
    for b_idx, e1_weight in izip(neighbour_idx[a_idx], weights_from[a_idx]):
        if filter_node(b_idx):
            continue
        for c_idx, e2_weight in izip(neighbour_idx[b_idx], weights_from[b_idx]):
            if c_exclude_idx[c_idx]:
                continue
            c_score[c_idx] = acc(c_score[c_idx], agg(e1_weight, e2_weight))
            c_is_c_idx[c_idx] = 1
