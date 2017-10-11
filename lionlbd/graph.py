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
from lionlbd.neo4jcsv import transpose


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
        self._nodes = nodes
        self._edges = edges

        self._tnodes = transpose(nodes)
        self._tedges = transpose(edges)

        self.min_year, self.max_year = edges[0].year, edges[-1].year
        info('min edge year {}, max edge year {}'.format(
            self.min_year, self.max_year))

        self.node_idx_by_id = self._create_node_idx_mapping(nodes)

        self.neighbours = self._create_neighbour_sequences(
            nodes, edges, self.node_idx_by_id)

        self.edges_from = self._create_edge_sequences(
            nodes, edges, self.node_idx_by_id)

        self.edge_metrics = self._get_metrics(edges[0]) if edges else []

        debug('initialized Graph: {}'.format(self.stats_str()))

    @timed
    def get_neighbours(self, id_, metric=None, types=None, year=None,
                       limit=None, indices_only=False):
        """Get neighbours of node.

        Args:
             indices_only: If True, only return neighbour indices.
        """
        try:
            idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))

        year = self._validate_year(year)

        filter_node = self._get_node_filter(types)
        filter_edge = self._get_edge_filter(year)
        get_score = self._get_edge_scorer(metric, year)

        scores, n_indices = [], []
        for n_idx, e_idx in izip(self.neighbours[idx], self.edges_from[idx]):
            if filter_node(n_idx) or filter_edge(e_idx):
                continue
            scores.append(get_score(e_idx))
            n_indices.append(n_idx)

        argsorted = reversed(argsort(scores))    # TODO: argpartition if limit?
        limit = limit if limit is not None else len(scores)

        if indices_only:
            return n_indices[:limit]    # TODO: return without sort?

        nodes = self._nodes
        results = []
        build_result = self._get_result_builder(degree=1, type_='lion')
        for i, idx in enumerate(argsorted, start=1):
            if i > limit:
                break
            results.append(build_result(n_indices[idx], scores[idx]))

        return results

    @timed
    def get_2nd_neighbours(self, id_, metric=None, types=None, year=None,
                           limit=None):
        """Get 2nd-degree neighbours of node.

        Excludes the starting node and its 1st-degree neighbours.
        """
        try:
            a_idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))

        year = self._validate_year(year)

        agg = self._get_agg_function('avg')    # TODO
        acc = self._get_acc_function('max')    # TODO

        filter_node = self._get_node_filter(types)
        filter_edge = self._get_edge_filter(year)
        get_score = self._get_edge_scorer(metric, year)

        nodes = self._nodes

        # Flag nodes to exclude for fast access in inner loop.
        exclude_idx = array('b', [0]) * len(nodes)
        # TODO: include constraints other than year?
        b_indices = self.get_neighbours(id_, year=year, indices_only=True)
        for b_idx in b_indices:
            exclude_idx[b_idx] = 1
        exclude_idx[a_idx] = 1

        # TODO: skip this loop if there is no node filter
        for i in range(len(nodes)):
            exclude_idx[i] |= filter_node(i)

        # accumulate scores by node in array
        score = array('f', [0]) * len(nodes)
        is_c_idx = array('b', [0]) * len(nodes)

        neighbours, edges_from = self.neighbours, self.edges_from
        for b_idx, e1_idx in izip(neighbours[a_idx], edges_from[a_idx]):
            if filter_node(b_idx) or filter_edge(e1_idx):
                continue
            e1_score = get_score(e1_idx)
            for c_idx, e2_idx in izip(neighbours[b_idx], edges_from[b_idx]):
                if (exclude_idx[c_idx] or filter_edge(e2_idx)):
                    continue
                e2_score = get_score(e2_idx)
                score[c_idx] = acc(score[c_idx], agg(e1_score, e2_score))
                is_c_idx[c_idx] = True

        argsorted = reversed(argsort(score))    # TODO: argpartition if limit?
        limit = limit if limit is not None else len(nodes)

        results = []
        build_result = self._get_result_builder(degree=2, type_='lion')
        for idx in argsorted:
            if len(results) >= limit:
                break
            if not is_c_idx[idx]:
                continue
            results.append(build_result(idx, score[idx]))
        return results

    def _validate_year(self, year):
        """Verify that given year is a valid value."""
        if year is None:
            return year
        if year < self.min_year or year > self.max_year:
            # TODO: bound instead of raise?
            raise ValueError('out of bounds year {}'.format(year))
        return year

    def _get_node_filter(self, types=None):
        """Return function determining whether to filter out a node."""
        if types is None:
            return lambda n_idx: False
        else:
            node_types, types = self._tnodes.type, set(types)
            return lambda n_idx: node_types[n_idx] not in types

    def _get_edge_filter(self, max_year=None):
        """Return function determining whether to filter out an edge."""
        if max_year is None or max_year == self.max_year:
            return lambda e_idx: False    # filter nothing
        else:
            edge_year = self._tedges.year
            return lambda e_idx: edges_year[e_idx] > max_year

    def _get_edge_scorer(self, metric=None, year=None):
        """Return function returning score for edge."""
        if metric is None:
            metric = 'count'    # TODO configurable default
        if year is None:
            year = self.max_year
        offset = year - self.max_year - 1

        idx_type = { m: (i, type_) for i, m, type_ in self.edge_metrics }
        if metric not in idx_type:
            raise ValueError('unknown metric {}'.format(metric))
        m_idx, m_type = idx_type[metric]

        edge_metrics = self._tedges[m_idx]
        return lambda e_idx: edge_metrics[e_idx][offset]

    def _get_result_builder(self, degree=1, type_='id'):
        """Return function for building result objects."""
        node_id = self._tnodes.id
        node_type = self._tnodes.type

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

    def stats_str(self):
        """Return Graph statistics as string."""
        return '{} nodes, {} edges'.format(len(self._nodes), len(self._edges))

    @staticmethod
    def _sort_nodes_and_edges(nodes, edges):
        """Sort node and edge arrays for efficient graph search."""

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
    def _create_neighbour_sequences(nodes, edges, idx_map):
        """Create index-based mapping from node to neighbouring nodes."""
        # TODO: consider arrays instead of lists
        neighbours = [[] for _ in xrange(len(nodes))]
        for edge in edges:
            start, end = idx_map[edge.start], idx_map[edge.end]
            neighbours[start].append(end)
        return neighbours

    @staticmethod
    def _create_edge_sequences(nodes, edges, idx_map):
        """Create index-based mapping from node to edges."""
        # TODO: consider arrays instead of lists
        edges_from = [[] for _ in xrange(len(nodes))]
        for idx, edge in enumerate(edges):
            start = idx_map[edge.start]
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
        # positions of the START_ID and END_ID values
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
    def _get_metrics(edge):
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
