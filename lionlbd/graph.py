#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

import pyximport; pyximport.install()
from lionlbd.cgraph import open_discovery_core, mark_type_filtered
from lionlbd.cgraph import reindex_float

from array import array
from itertools import izip
from bisect import bisect
from logging import debug, info, warn, error

from numpy import argsort

from lionlbd.config import METRIC_PREFIX, METRIC_SUFFIX
from lionlbd.common import timed
from lionlbd.neo4jcsv import transpose, array_type_code
from lionlbd.lbdinterface import LbdFilters


class Graph(object):
    """In-memory Neo4j graph."""

    Filters = LbdFilters

    def __init__(self, nodes, edges):
        """Initialize Graph.

        Note: sorts given lists of nodes and edges.
        """
        if not nodes:
            raise ValueError('no nodes')
        if not edges:
            raise ValueError('no edges')

        nodes, edges = self._to_directed_graph(nodes, edges)
        self._node_count = len(nodes)
        self._edge_count = len(edges)

        self._sort_nodes_and_edges(nodes, edges)

        self._node_idx_by_id = self._create_node_idx_mapping(nodes)

        self._min_year, self._max_year = edges[0].year, edges[-1].year
        info('min edge year {}, max edge year {}'.format(
            self._min_year, self._max_year))

        self._metrics = self._get_edge_metrics(edges[0])

        self._nodes_t = transpose(nodes)
        self._edges_t = transpose(edges)
        nodes, edges = None, None    # release

        self._node_type_map = self._create_binary_type_map(self._nodes_t)
        self._nodes_t = self._map_types(self._nodes_t, self._node_type_map)

        self._edges_t = self._ids_to_indices(
            self._edges_t, self._node_idx_by_id)

        self._weights_by_metric_and_year = {}
        for m_idx, m_name, m_type in self._metrics:
            self._weights_by_metric_and_year[m_name] = self._group_by_year(
                m_name, m_type, self._edges_t[m_idx], self._edges_t.year,
                self._min_year, self._max_year)
            self._edges_t = self._remove_metric(
                m_name, self._edges_t, self._metrics)

        self._neighbour_idx = self._create_neighbour_sequences(
            self._node_count, self._edges_t)

        # TODO is this needed?
        self._edges_from = self._create_edge_sequences(
            self._node_count, self._edges_t)

        self._weights_from_cache = {}    # lazy init
        # for year in range(self._min_year, self._max_year+1):
        #     self._get_weights_from('count', year)    # precache (TODO)
        self._get_weights_from('count', self._max_year)    # precache (TODO)

        debug('initialized Graph: {}'.format(self.stats_str()))

    @timed
    def neighbours(self, id_, metric, year=None, filters=None, limit=None,
                   offset=0, indices_only=False):
        """Return neighbours of node.

        Args:
             indices_only: If True, only return neighbour indices.
        """
        idx = self._get_node_idx(id_)

        metric = self._validate_metric(metric)
        year = self._validate_year(year)
        limit = self._validate_limit(limit)
        offset = self._validate_offset(offset)
        # TODO: validate filters

        if filters:
            filter_node = self._get_node_filter(filters.b_types)
        else:
            filter_node = self._get_node_filter(None)

        scores, n_indices = [], []
        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)
        for n_idx, e_weight in izip(neighbour_idx[idx], weights_from[idx]):
            if filter_node(n_idx):
                continue
            scores.append(e_weight)
            n_indices.append(n_idx)

        argsorted = reversed(argsort(scores))    # TODO: argpartition if limit?
        end_idx = offset+limit if limit is not None else len(scores)

        if indices_only:
            return n_indices[offset:end_idx]    # TODO: return without sort?

        results = []
        build_result = self._get_result_builder(degree=1, type_='lion')
        for i, idx in enumerate(argsorted, start=1):
            if i > end_idx:
                break
            if i > offset:
                results.append(build_result(n_indices[idx], scores[idx]))

        return results

    @timed
    def open_discovery(self, a_id, metric, agg_func, acc_func, year=None,
                       filters=None, limit=None, offset=0):
        """Get 2nd-degree neighbours of node.

        Excludes the starting node and its 1st-degree neighbours.
        """
        a_idx = self._get_node_idx(a_id)

        metric = self._validate_metric(metric)
        year = self._validate_year(year)
        limit = self._validate_limit(limit)
        offset = self._validate_offset(offset)
        # TODO: validate filters, agg_func and acc_func

        node_count = self._node_count

        # Flag nodes to exclude for fast access in inner loop.
        exclude_idx = array('b', [0]) * node_count
        # TODO: include constraints other than year?
        b_indices = self.neighbours(a_id, metric, year=year, indices_only=True)
        for b_idx in b_indices:
            exclude_idx[b_idx] = 1
        exclude_idx[a_idx] = 1

        if filters:
            mark_type_filtered(exclude_idx, self._nodes_t.type, filters.c_types,
                               self._node_type_map)

        # accumulate scores by node in array
        score = array('f', [0]) * node_count
        is_c_idx = array('b', [0]) * node_count

        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)

        if filters:
            filter_b_node = self._get_node_filter(filters.b_types)
        else:
            filter_b_node = self._get_node_filter(None)

        open_discovery_core(a_idx, neighbour_idx, weights_from, score,
                            is_c_idx, exclude_idx, filter_b_node,
                            agg_func, acc_func)

        argsorted = reversed(argsort(score))    # TODO: argpartition if limit?
        limit = limit if limit is not None else node_count
        end_idx = offset+limit if limit is not None else len(scores)

        results, result_idx = [], 0
        build_result = self._get_result_builder(degree=2, type_='lion')
        for idx in argsorted:
            if len(results) >= limit:
                break
            if not is_c_idx[idx]:
                continue
            if result_idx >= offset:
                results.append(build_result(idx, score[idx]))
            result_idx += 1
        return results

    def get_metrics(self):
        """Return edge weight metrics used in the graph.

        Returns:
            list of str: edge metric names.
        """
        return [name for index, name, type_ in self._metrics]

    def _get_node_idx(self, id_):
        try:
            idx = self._node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))
        return idx

    def _metric_type(self, metric):
        """Return type of values for metric (e.g. float)."""
        for index, name, type_ in self._metrics:
            if name == metric:
                return type_
        raise ValueError(metric)

    def _validate_year(self, year):
        """Verify that given year is valid, apply default if None."""
        if year is None:
            return self._max_year
        elif year < self._min_year or year > self._max_year:
            raise ValueError('out of bounds year {}'.format(year))
        return year

    def _validate_limit(self, limit):
        """Verify that given limit is valid."""
        if limit is None:
            return limit    # None is valid
        elif limit <= 0:
            raise ValueError('out of bounds limit {}'.format(limit))
        return limit

    def _validate_offset(self, offset):
        """Verify that given offset is valid, apply default if None"""
        if offset is None:
            return 0
        elif offset < 0:
            raise ValueError('out of bounds offset {}'.format(offset))
        return offset

    def _validate_metric(self, metric):
        """Verify that given metric is valid, apply default if None."""
        if metric is None:
            return 'count'    # TODO configurable default
        elif metric in self.get_metrics():
            return metric
        else:
            raise ValueError('invalid metric {}'.format(metric))

    def _get_node_filter(self, types=None):
        """Return function determining whether to filter out a node."""
        if types is None:
            return lambda n_idx: False
        else:
            type_bits = [self._node_type_map[t] for t in types]
            type_mask = reduce(lambda x,y: x|y, type_bits)
            # debug('type_mask {} for {}'.format(type_mask, types))
            node_types = self._nodes_t.type
            return lambda n_idx: not (node_types[n_idx] & type_mask)

    def _get_result_builder(self, degree=1, type_='id'):
        """Return function for building result objects."""
        node_id = self._nodes_t.id
        node_type = self._nodes_t.type
        inv_type_map = { v: k for k, v in self._node_type_map.items() }

        def id_only(n_idx, *args):
            return node_id[n_idx]

        def lion_1st(n_idx, score, *args):
            return {
                'B': node_id[n_idx],
                'B_type': inv_type_map[node_type[n_idx]],
                'comp': score,
            }

        def lion_2nd(n_idx, score, *args):
            return {
                'C': node_id[n_idx],
                'C_type': inv_type_map[node_type[n_idx]],
                'comp': score,
            }

        if type_ == 'id':
            return id_only
        elif degree == 1:    # 1st degree (immediate) neighbours
            if type_ == 'lion':
                return lion_1st
            else:
                raise NotImplementedError('{}/{}'.format(degree, type_))
        elif degree == 2:
            if type_ == 'lion':
                return lion_2nd
            else:
                raise NotImplementedError('{}/{}'.format(degree, type_))
        else:
            raise NotImplementedError('{}/{}'.format(degree, type_))

    @timed
    def _get_weights_from(self, metric, year):
        if metric not in self._weights_from_cache:
            self._weights_from_cache[metric] = {}
        if year not in self._weights_from_cache[metric]:
            # lazy init
            info('calculating weights_from for metric {}, year {} ...'.format(
                metric, year))
            edge_weight = self._weights_by_metric_and_year[metric][year]
            type_code = array_type_code(self._metric_type(metric))
            idx_after = bisect(self._edges_t.year, year)    # sorted by year
            self._weights_from_cache[metric][year] = reindex_float(
                self._node_count, self._edges_t.start, edge_weight, idx_after)
        return self._weights_from_cache[metric][year]

    def stats_str(self):
        """Return Graph statistics as string."""
        return '{} nodes, {} edges'.format(self._node_count, self._edge_count)

    @staticmethod
    def _sort_nodes_and_edges(nodes, edges):
        """Sort node and edge arrays for efficient storage and search."""

        # sort edges by year
        edges.sort(key=lambda e: e.year)

        # for each node, identify earliest year with an edge
        first_edge_year = {}
        for edge in edges:
            for n_id in (edge.start, edge.end):
                if n_id not in first_edge_year:
                    first_edge_year[n_id] = edge.year
        max_year = edges[-1].year
        edgeless = [n for n in nodes if n.id not in first_edge_year]
        if edgeless:
            ids = [n.id for n in edgeless]
            id_str = (', '.join(ids) if len(ids) < 5 else
                      ', '.join(ids[:5]) + '...')
            warn('{} nodes without edges ({})'.format(len(edgeless), ids))
        for node in edgeless:
            # sort in after other nodes, otherwise arbitrary order
            first_edge_year[node.id] = max_year + 1

        # sort nodes by year of earliest edge
        nodes.sort(key=lambda n: first_edge_year[n.id])

    @staticmethod
    def _create_node_idx_mapping(nodes):
        """Create mapping from node ID to index in given sequence."""
        node_idx_by_id = {}
        for idx, node in enumerate(nodes):
            if node.id in node_idx_by_id:
                raise ValueError('duplicate node ID {}'.format(node.id))
            node_idx_by_id[node.id] = idx
        return node_idx_by_id

    @staticmethod
    def _create_binary_type_map(nodes_t):
        """Create binary encoding of node types.

        Note:
            Expects nodes_t to be result of transpose(nodes).
        """
        types = sorted(list(set(nodes_t.type)))
        info('types: {}'.format(types))
        type_map = { t: 1<<i for i, t in enumerate(types) }
        debug('binary type mapping: {}'.format(type_map))
        return type_map

    @staticmethod
    def _map_types(nodes_t, type_map):
        """Replace node types with values from given map.

        Note:
            Expects nodes_t to be result of transpose(nodes).
        """
        nodes_d = nodes_t._asdict()
        nodes_d['type'] = array('i', (type_map[t] for t in nodes_t.type))
        class_ = type(nodes_t)
        return class_(**nodes_d)

    @staticmethod
    def _ids_to_indices(edges_t, node_idx_by_id):
        """Replace node ids with indices in edge data.

        Note:
            Expects edges_t to be result of transpose(edges).
        """
        edges_d = edges_t._asdict()
        edges_d['start'] = array('i',(node_idx_by_id[i] for i in edges_t.start))
        edges_d['end'] = array('i', (node_idx_by_id[i] for i in edges_t.end))
        class_ = type(edges_t)
        return class_(**edges_d)

    @staticmethod
    @timed
    def _group_by_year(name, m_type, edge_weights, edge_years, min_year,
                       max_year):
        """Group metric values by year.

        Note:
            Expects edges_t to be result of transpose(edges) and sorted by year.
        """
        type_code = array_type_code(m_type)
        if type_code != 'f':
            warn('converting weight type {} to float'.format(type_code))
            type_code = 'f'
        weights_by_year = {}
        for year in range(min_year, max_year+1):
            y_idx = year - max_year - 1
            idx_limit = bisect(edge_years, year)    # edges sorted by year
            weights_by_year[year] = array(
                type_code, (edge_weights[i][y_idx] for i in range(idx_limit)))
        return weights_by_year

    @staticmethod
    def _remove_metric(name, edges_t, metrics):
        """Remove metric from transposed edges.

        Use with _group_by_year(), avoids storing metrics redundantly.
        """
        edges_l = list(edges_t)
        for m_idx, m_name, m_type in metrics:
            if m_name == name:
                edges_l[m_idx] = None
        class_ = type(edges_t)
        return class_(*edges_l)

    @staticmethod
    def _create_neighbour_sequences(node_count, edges_t):
        """Create index-based mapping from node to neighbouring nodes.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        neighbour_idx = [array('i') for _ in xrange(node_count)]
        for start, end in izip(edges_t.start, edges_t.end):
            neighbour_idx[start].append(end)
        return neighbour_idx

    @staticmethod
    def _create_edge_sequences(node_count, edges_t):
        """Create index-based mapping from node to edges.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        edges_from = [array('i') for _ in xrange(node_count)]
        for idx, start in enumerate(edges_t.start):
            edges_from[start].append(idx)
        return edges_from

    @staticmethod
    def _to_directed_graph(nodes, edges):
        """Given an undirected graph, return a directed equivalent.

        Note: produces shallow copies of edges, leading to referenced
        objects beings shared between edges.
        """
        if not edges:
            return nodes, edges

        # assume edges are namedtuples and all edges share the index
        # positions of the start and end values
        fields = type(edges[0])._fields
        START, END = fields.index('start'), fields.index('end')

        directed_edges = edges[:]
        for edge in edges:
            e_class = type(edge)
            e_list = list(edge)
            e_list[START], e_list[END] = e_list[END], e_list[START]
            directed_edges.append(e_class(*e_list))
        return nodes, directed_edges

    @staticmethod
    def _get_edge_metrics(edge):
        """Return list of metric (index, name, type) tuples for namedtuple."""
        metrics = []
        for index, field in enumerate(type(edge)._fields):
            if (field.startswith(METRIC_PREFIX) and
                field.endswith(METRIC_SUFFIX)):
                value = edge[index]
                if METRIC_SUFFIX:
                    name = field[len(METRIC_PREFIX):-len(METRIC_SUFFIX)]
                else:
                    name = field[len(METRIC_PREFIX):]
                try:
                    type_ = type(value[0])    # assume sequence
                except:
                    error('expected sequence of values, got {}'.format(
                        type(value)))
                    raise
                metrics.append((index, name, type_))
        if not metrics:
            warn('no metrics found for {}'.format(edge))
        else:
            debug('metrics for {}({}): {}'.format(
                type(edge).__name__, type(edge)._fields, metrics))
        return metrics

    @staticmethod
    def _get_agg_function(name):
        if name == 'sum':
            return lambda a, b: a + b
        elif name == 'avg':
            return lambda a, b: (a + b) / 2.0
        else:
            raise NotImplementedError(name)

    @staticmethod
    def _get_acc_function(name):
        if name == 'max':
            return lambda a, b: max(a, b)
        if name == 'sum':
            return lambda a, b: a + b
        else:
            raise NotImplementedError(name)
