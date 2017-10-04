#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

from itertools import izip
from logging import debug, info, warn, error


class Graph(object):
    """In-memory Neo4j graph."""

    def __init__(self, nodes, edges):
        """Initialize Graph."""
        if not nodes:
            raise ValueError('no nodes')
        if not edges:
            raise ValueError('no edges')

        self._nodes = nodes
        self._edges = edges
        debug('initialized Graph: {}'.format(self.stats_str()))

        self.node_idx_by_id = self._create_node_idx_mapping(nodes)

        self.neighbours = self._create_neighbour_sequences(
            nodes, edges, self.node_idx_by_id)

        self.edges_from = self._create_edge_sequences(
            nodes, edges, self.node_idx_by_id)

    def get_neighbours(self, id_):
        """Get neighbours for given node."""
        try:
            idx = self.node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))
        result = []
        nodes, edges = self._nodes, self._edges
        for n_idx, e_idx in izip(self.neighbours[idx], self.edges_from[idx]):
            result.append(nodes[n_idx])
        return result

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
