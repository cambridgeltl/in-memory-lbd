#!/usr/bin/env python

# Application startup functionality

from __future__ import print_function
from __future__ import absolute_import

import logging

from lionlbd.config import NODE_FILE, EDGE_FILE
from lionlbd.neo4jcsv import load_nodes, load_edges
from lionlbd.graph import Graph
from lionlbd.common import timed, memory_usage


def parse_args():
    import sys
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--nodes', metavar='FILE', default=NODE_FILE,
                    help='Node CSV')
    ap.add_argument('-e', '--edges', metavar='FILE', default=EDGE_FILE,
                    help='Edge CSV')
    return ap.parse_args(sys.argv[1:])

args = parse_args()

logging.debug('node file {}'.format(args.nodes))
logging.debug('edge file {}'.format(args.edges))

@timed
def load_graph(node_file, edge_file):
    return Graph(load_nodes(node_file), load_edges(edge_file))

graph = load_graph(args.nodes, args.edges)

logging.info('graph loaded, using {} memory'.format(memory_usage('M')))
