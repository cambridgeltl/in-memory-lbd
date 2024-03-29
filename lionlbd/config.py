import os

# Flask debug mode
DEBUG = True

# Port to listen on
PORT = 8081

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

# Metadata to load on startup
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

# Parameter determining type
FILTER_TYPE_PARAMETER = 'type'

# Parameter determining (max) year
YEAR_PARAMETER = 'year'

# Parameter determining "until" year for discoverable edges
UNTIL_PARAMETER = 'until'

# Parameter determining edge scoring metric
METRIC_PARAMETER = 'edge_metric'

# Parameter determining maximum number of results to return
LIMIT_PARAMETER = 'limit'

# Parameter determining offset of first result to return
OFFSET_PARAMETER = 'offset'

# Naming convention for metrics in edge data
METRIC_PREFIX = 'metric_'
METRIC_SUFFIX = ''
