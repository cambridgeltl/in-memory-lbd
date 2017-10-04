from __future__ import print_function
from __future__ import absolute_import

import logging

from lionlbd import app
from lionlbd import graph


@app.route('/')
@app.route('/index')
def index():
    return '{} nodes, {} edges'.format(len(graph[0][1]), len(graph[1][1]))
