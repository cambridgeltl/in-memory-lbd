#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

from logging import debug, info, warn, error


class Graph(object):
    def __init__(self, nodes, edges):
        """Initialize Graph."""
        self._nodes = nodes
        self._edges = edges

    def stats_str(self):
        """Return Graph statistics as string."""
        return '{} nodes, {} edges'.format(len(self._nodes), len(self._edges))
