#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

from array import array
from itertools import izip
from logging import debug, info, warn, error

from numpy import argsort

from lionlbd.config import METRIC_PREFIX, METRIC_SUFFIX
from lionlbd.common import timed
from lionlbd.neo4jcsv import transpose, array_type_code


class Graph(object):
    """In-memory Neo4j graph."""

    def __init__(self, nodes, edges):
        """Initialize Graph.

        Note: sorts given lists of nodes and edges.
        """
        if not nodes:
            raise ValueError('no nodes')
        if not edges:
            raise ValueError('no edges')

        nodes, edges = self._to_directed_graph(nodes, edges)

        self._sort_nodes_and_edges(nodes, edges)

        self.node_idx_by_id = self._create_node_idx_mapping(nodes)

        self.min_year, self.max_year = edges[0].year, edges[-1].year
        info('min edge year {}, max edge year {}'.format(
            self.min_year, self.max_year))

        self.edge_metrics = self._get_edge_metrics(edges[0])

        self._nodes_t = transpose(nodes)
        self._edges_t = transpose(edges)
        self._edges_t = self._ids_to_indices(self._edges_t, self.node_idx_by_id)

        self._weights_by_metric_and_year = self._group_by_year(
            self._edges_t, self.edge_metrics)

        self._neighbour_idx = self._create_neighbour_sequences(
            len(nodes), self._edges_t)

        # TODO is this needed?
        self.edges_from = self._create_edge_sequences(
            len(nodes), self._edges_t)

        self.weights_from_cache = {}    # lazy init
        self._get_weights_from('count', self.max_year)    # precache (TODO)

        debug('initialized Graph: {}'.format(self.stats_str()))

    @timed
    def neighbours(self, id_, metric=None, types=None, year=None,
                   limit=None, indices_only=False):
        """Return neighbours of node.

        Args:
             indices_only: If True, only return neighbour indices.
        """
        try:
            idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))

        metric = self._validate_metric(metric)
        year = self._validate_year(year)

        filter_node = self._get_node_filter(types)

        scores, n_indices = [], []
        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)
        for n_idx, e_weight in izip(neighbour_idx[idx], weights_from[idx]):
            if filter_node(n_idx):
                continue
            scores.append(e_weight)
            n_indices.append(n_idx)

        argsorted = reversed(argsort(scores))    # TODO: argpartition if limit?
        limit = limit if limit is not None else len(scores)

        if indices_only:
            return n_indices[:limit]    # TODO: return without sort?

        results = []
        build_result = self._get_result_builder(degree=1, type_='lion')
        for i, idx in enumerate(argsorted, start=1):
            if i > limit:
                break
            results.append(build_result(n_indices[idx], scores[idx]))

        return results

    @timed
    def open_discovery(self, id_, metric=None, types=None, year=None,
                       limit=None):
        """Get 2nd-degree neighbours of node.

        Excludes the starting node and its 1st-degree neighbours.
        """
        try:
            a_idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))

        metric = self._validate_metric(metric)
        year = self._validate_year(year)

        agg = self._get_agg_function('avg')    # TODO
        acc = self._get_acc_function('max')    # TODO

        filter_node = self._get_node_filter(types)

        node_count = len(self._nodes_t.id)

        # Flag nodes to exclude for fast access in inner loop.
        exclude_idx = array('b', [0]) * node_count
        # TODO: include constraints other than year?
        b_indices = self.neighbours(id_, year=year, indices_only=True)
        for b_idx in b_indices:
            exclude_idx[b_idx] = 1
        exclude_idx[a_idx] = 1

        # TODO: skip this loop if there is no node filter
        for i in range(node_count):
            exclude_idx[i] |= filter_node(i)

        # accumulate scores by node in array
        score = array('f', [0]) * node_count
        is_c_idx = array('b', [0]) * node_count

        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)
        for b_idx, e1_weight in izip(neighbour_idx[a_idx], weights_from[a_idx]):
            if filter_node(b_idx):
                continue
            for c_idx, e2_weight in izip(neighbour_idx[b_idx], weights_from[b_idx]):
                if exclude_idx[c_idx]:
                    continue
                score[c_idx] = acc(score[c_idx], agg(e1_weight, e2_weight))
                is_c_idx[c_idx] = True

        argsorted = reversed(argsort(score))    # TODO: argpartition if limit?
        limit = limit if limit is not None else node_count

        results = []
        build_result = self._get_result_builder(degree=2, type_='lion')
        for idx in argsorted:
            if len(results) >= limit:
                break
            if not is_c_idx[idx]:
                continue
            results.append(build_result(idx, score[idx]))
        return results

    def get_metrics(self):
        """Return edge weight metrics used in the graph.

        Returns:
            list of str: edge metric names.
        """
        return [name for index, name, type_ in self.edge_metrics]

    def _validate_year(self, year):
        """Verify that given year is valid, apply default if None."""
        if year is None:
            return self.max_year
        if year < self.min_year or year > self.max_year:
            raise ValueError('out of bounds year {}'.format(year))
        return year

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
            node_types, types = self._nodes_t.type, set(types)
            return lambda n_idx: node_types[n_idx] not in types

    def _get_result_builder(self, degree=1, type_='id'):
        """Return function for building result objects."""
        node_id = self._nodes_t.id
        node_type = self._nodes_t.type

        def id_only(n_idx, *args):
            return node_id[n_idx]

        def lion_1st(n_idx, score, *args):
            return {
                'B': node_id[n_idx],
                'B_type': node_type[n_idx],
                'comp': score,
            }

        def lion_2nd(n_idx, score, *args):
            return {
                'C': node_id[n_idx],
                'C_type': node_type[n_idx],
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
        if metric not in self.weights_from_cache:
            self.weights_from_cache[metric] = {}
        if year not in self.weights_from_cache[metric]:
            # lazy init
            info('calculating weights_from for metric {}, year {} ...'.format(
                metric, year))
            # idx_map = self.node_idx_by_id
            weights_by_idx = self._weights_by_metric_and_year[metric][year]
            node_count = len(self._nodes_t.id)
            weights_from = [[] for _ in xrange(node_count)]
            for idx, start in enumerate(self._edges_t.start):
                if self._edges_t.year[idx] > year:
                    break    # edges sorted by year
                # start = idx_map[edge.start]
                weights_from[start].append(weights_by_idx[idx])
            self.weights_from_cache[metric][year] = weights_from
        return self.weights_from_cache[metric][year]

    def stats_str(self):
        """Return Graph statistics as string."""
        node_count = len(self._nodes_t.id)
        edge_count = len(self._edges_t.start)
        return '{} nodes, {} edges'.format(node_count, edge_count)

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
    def _ids_to_indices(edges_t, node_idx_by_id):
        """Replace node ids with indices in edge data.

        Note:
            Expects edges_t to be result of transpose(edges).
        """
        edges_d = edges_t._asdict()
        edges_d['start'] = array('i',[node_idx_by_id[i] for i in edges_t.start])
        edges_d['end'] = array('i', [node_idx_by_id[i] for i in edges_t.end])
        class_ = type(edges_t)
        return class_(**edges_d)

    @staticmethod
    def _group_by_year(edges_t, edge_metrics):
        """Group metric values by year.

        Note:
            Expects edges_t to be result of transpose(edges) and sorted by year.
        """
        min_year, max_year = edges_t.year[0], edges_t.year[-1]
        weights_by_metric_and_year = {}
        for m_idx, m_name, m_type in edge_metrics:
            weights_by_year = {}
            weights_by_edge_and_year = edges_t[m_idx]
            for year in range(min_year, max_year+1):
                weights = []
                year_idx = year - max_year - 1
                for e_idx, e_year in enumerate(edges_t.year):
                    if e_year > year:
                        break    # edges sorted by year
                    weights.append(weights_by_edge_and_year[e_idx][year_idx])
                weights_by_year[year] = array(
                    array_type_code(m_type), weights)
            weights_by_metric_and_year[m_name] = weights_by_year
        return weights_by_metric_and_year

    @staticmethod
    def _create_neighbour_sequences(node_count, edges_t):
        """Create index-based mapping from node to neighbouring nodes.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        neighbour_idx = [[] for _ in xrange(node_count)]
        for start, end in izip(edges_t.start, edges_t.end):
            neighbour_idx[start].append(end)
        return neighbour_idx

    @staticmethod
    def _create_edge_sequences(node_count, edges_t):
        """Create index-based mapping from node to edges.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        edges_from = [[] for _ in xrange(node_count)]
        for idx, start in enumerate(edges_t.start):
            edges_from[start].append(idx)
        edges_from = [array('i', i) for i in edges_from]
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
                name = field[len(METRIC_PREFIX):-len(METRIC_SUFFIX)]
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
