import os

# Flask debug mode
DEBUG = True

# Port to listen on
PORT = 8080

# http://jinja.pocoo.org/docs/templates/#whitespace-control
TRIM_BLOCKS = True

# https://stackoverflow.com/a/3718923
CONFIG_DIR =  os.path.dirname(os.path.abspath(__file__))

# Directory with node and edge data
DATA_DIR = os.path.join(CONFIG_DIR, '../data/neoplasms-1p')

# Node data to load on startup
NODE_FILE = os.path.join(DATA_DIR, 'nodes.csv')

# Edge data to load on startup
EDGE_FILE = os.path.join(DATA_DIR, 'edges.csv')
