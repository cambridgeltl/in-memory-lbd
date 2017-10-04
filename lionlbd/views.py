from __future__ import print_function
from __future__ import absolute_import

import logging

from lionlbd import app
from lionlbd import graph


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()
