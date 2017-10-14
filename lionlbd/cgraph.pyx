from __future__ import print_function
from __future__ import absolute_import

cimport cython

from cpython cimport array

import array

from itertools import izip


def open_discovery_core(int a_idx,
                        list neighbour_idx,
                        list weights_from,
                        array.array score,
                        array.array is_c_idx,
                        array.array exclude_idx,
                        filter_node,
                        agg, acc):
    
    for b_idx, e1_weight in izip(neighbour_idx[a_idx], weights_from[a_idx]):
        if filter_node(b_idx):
            continue
        for c_idx, e2_weight in izip(neighbour_idx[b_idx], weights_from[b_idx]):
            if exclude_idx[c_idx]:
                continue
            score[c_idx] = acc(score[c_idx], agg(e1_weight, e2_weight))
            is_c_idx[c_idx] = True
