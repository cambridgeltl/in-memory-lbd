#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

from itertools import izip
from logging import debug, info, warn, error

from numpy import argsort

from lionlbd.config import METRIC_PREFIX, METRIC_SUFFIX


class Graph(object):
    """In-memory Neo4j graph."""

    def __init__(self, nodes, edges):
        """Initialize Graph."""
        if not nodes:
            raise ValueError('no nodes')
        if not edges:
            raise ValueError('no edges')

        nodes, edges = self._to_directed_graph(nodes, edges)

        self._nodes = nodes
        self._edges = edges
        debug('initialized Graph: {}'.format(self.stats_str()))

        self.node_idx_by_id = self._create_node_idx_mapping(nodes)

        self.neighbours = self._create_neighbour_sequences(
            nodes, edges, self.node_idx_by_id)

        self.edges_from = self._create_edge_sequences(
            nodes, edges, self.node_idx_by_id)

        self.max_year = max(e.year for e in edges)

        self.edge_metrics = self._get_metrics(edges[0]) if edges else []

    def get_neighbours(self, id_, metric=None, types=None, year=None,
                       limit=None):
        """Get neighbours for given node."""
        try:
            idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))

        nodes, edges = self._nodes, self._edges
        filter_node = self._get_node_filter(types)
        filter_edge = self._get_edge_filter(year)
        get_score = self._get_edge_scorer(metric, year)

        scores, n_indices = [], []
        for n_idx, e_idx in izip(self.neighbours[idx], self.edges_from[idx]):
            if not filter_node(n_idx) and not filter_edge(e_idx):
                scores.append(get_score(e_idx))
                n_indices.append(n_idx)

        argsorted = reversed(argsort(scores))    # TODO: argpartition if limit?
        limit = limit if limit is not None else len(scores)
        results = []
        for i, idx in enumerate(argsorted, start=1):
            if i > limit:
                break
            results.append({
                'score': scores[idx],
                'B': nodes[n_indices[idx]].id
            })

        return results

    def _get_node_filter(self, types=None):
        """Return function determining whether to filter out a node."""
        if types is None:
            return lambda n_idx: False
        else:
            nodes, types = self._nodes, set(types)
            return lambda n_idx: nodes[n_idx].type not in types

    def _get_edge_filter(self, max_year=None):
        """Return function determining whether to filter out an edge."""
        if max_year is None:
            return lambda e_idx: False
        else:
            edges = self._edges
            return lambda e_idx: edges[e_idx].year > max_year

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

        edges = self._edges
        return lambda e_idx: edges[e_idx][m_idx][offset]

    def stats_str(self):
        """Return Graph statistics as string."""
        return '{} nodes, {} edges'.format(len(self._nodes), len(self._edges))

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
