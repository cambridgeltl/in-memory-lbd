#!/usr/bin/env python

# Application startup functionality

from __future__ import print_function
from __future__ import absolute_import

import logging

from lionlbd.config import NODE_FILE, EDGE_FILE
from lionlbd.neo4jcsv import load_nodes, load_edges
from lionlbd.graph import Graph

logging.debug('node file {}'.format(NODE_FILE))
logging.debug('edge file {}'.format(EDGE_FILE))


graph = Graph(load_nodes(NODE_FILE), load_edges(EDGE_FILE))
