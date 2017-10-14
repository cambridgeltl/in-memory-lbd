from __future__ import print_function
from __future__ import absolute_import

cimport cython

from cpython cimport array

import array

from itertools import izip

from lionlbd.common import timed


# Temporarily use ints for scores
ctypedef int score_t


# Type of aggregation and accumulation functions
ctypedef float (*agg_func_t)(float e1_score, float e2_score)
ctypedef float (*acc_func_t)(float accumulated, float score)


#@cython.boundscheck(False)
#@cython.wraparound(False)
@timed
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

    cdef agg_func_t c_agg = _get_agg_function(agg)
    cdef agg_func_t c_acc = _get_acc_function(acc)

    for b_idx, e1_weight in izip(neighbour_idx[a_idx], weights_from[a_idx]):
        if filter_node(b_idx):
            continue
        for c_idx, e2_weight in izip(neighbour_idx[b_idx], weights_from[b_idx]):
            if c_exclude_idx[c_idx]:
                continue
            #c_score[c_idx] = acc(c_score[c_idx], agg(e1_weight, e2_weight))
            c_score[c_idx] = c_acc(c_score[c_idx], c_agg(e1_weight, e2_weight))
            c_is_c_idx[c_idx] = 1



cdef float _agg_func_avg(float e1_score, float e2_score):
    return (e1_score + e2_score) / 2.0

cdef agg_func_t _get_agg_function(name):
    if name == 'avg':
        return _agg_func_avg
    else:
        raise ValueError('unknown aggregation function {}'.format(name))


cdef float _acc_func_max(float accumulated, float score):
    return accumulated if accumulated > score else score

cdef acc_func_t _get_acc_function(name):
    if name == 'max':
        return _acc_func_max
    else:
        raise ValueError('unknown accumulation function {}'.format(name))
