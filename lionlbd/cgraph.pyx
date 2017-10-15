from __future__ import print_function
from __future__ import absolute_import

cimport cython

from cpython cimport array

import array

from itertools import izip

from lionlbd.common import timed


ctypedef float score_t


# Type of aggregation and accumulation functions
ctypedef float (*agg_func_t)(float e1_score, float e2_score)
ctypedef float (*acc_func_t)(float accumulated, float score)


@timed
@cython.wraparound(False)
@cython.boundscheck(False)
def open_discovery_core(int a_idx,
                        list neighbour_idx,
                        list weights_from,
                        array.array score,
                        array.array is_c_idx,
                        array.array exclude_idx,
                        filter_b_node,
                        agg, acc):
    cdef int b_idx, c_idx
    cdef score_t e1_weight, e2_weight

    cdef float[:] c_score = score
    cdef signed char[:] c_is_c_idx = is_c_idx
    cdef signed char[:] c_exclude_idx = exclude_idx

    cdef agg_func_t c_agg = _get_agg_function(agg)
    cdef agg_func_t c_acc = _get_acc_function(acc)

    cdef int i1, i2, i1_limit, i2_limit

    i1_limit = min((len(neighbour_idx[a_idx]), len(weights_from[a_idx])))
    for i1 in xrange(i1_limit):
        b_idx = neighbour_idx[a_idx][i1]
        if filter_b_node(b_idx):
            continue
        e1_weight = weights_from[a_idx][i1]
        i2_limit = min((len(neighbour_idx[b_idx]), len(weights_from[b_idx])))
        for i2 in xrange(i2_limit):
            c_idx = neighbour_idx[b_idx][i2]
            if c_exclude_idx[c_idx]:
                continue
            e2_weight = weights_from[b_idx][i2]
            c_score[c_idx] = c_acc(c_score[c_idx], c_agg(e1_weight, e2_weight))
            c_is_c_idx[c_idx] = 1


@timed
def mark_type_filtered(array.array is_excluded, array.array node_type,
                       types, type_map):
    """Mark nodes whose types are not in the given list."""
    if types is None:
        return
    if not set(type_map.keys()) - set(types):
        return    # all types included

    cdef signed char[:] c_is_excluded = is_excluded
    cdef int[:] c_node_type = node_type
    cdef int type_mask = reduce(lambda x,y: x|y, (type_map[t] for t in types))
    cdef int i
    for i in range(len(is_excluded)):
        is_excluded[i] |= not (c_node_type[i] & type_mask)


# TODO: combine reindex_int and reindex_float using fused types
# (http://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html)

@timed
def reindex_int(int idx_count, int[:] idx_seq, int[:] val_seq, int limit):
    """Return a such that a[i] holds val_seq[j] where idx_seq[j] == i.

    Used e.g. to organize edge weights into lists indexed by the
    starting node index.
    """
    cdef int i
    cdef list reindexed = [array.array('i') for _ in xrange(idx_count)]
    for i in xrange(limit):
        reindexed[idx_seq[i]].append(val_seq[i])
    return reindexed


@timed
def reindex_float(int idx_count, int[:] idx_seq, float[:] val_seq, int limit):
    """Return a such that a[i] holds val_seq[j] where idx_seq[j] == i.

    Used e.g. to organize edge weights into lists indexed by the
    starting node index.
    """
    cdef int i
    cdef list reindexed = [array.array('f') for _ in xrange(idx_count)]
    for i in xrange(limit):
        reindexed[idx_seq[i]].append(val_seq[i])
    return reindexed


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
