from __future__ import print_function
from __future__ import absolute_import

import logging

from flask import abort, jsonify
from logging import debug, info, warn, error

from lionlbd import app
from lionlbd import graph
from lionlbd.common import get_metric, get_year, get_filter_type


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()


@app.route('/neighbours/<id_>')
def get_neighbours(id_):
    metric = get_metric()
    year = get_year()
    types = get_filter_type()
    try:
        result = graph.get_neighbours(
            id_,
            metric=metric,
            year=year,
            types=types
        )
    except KeyError, e:
        warn(e)
        abort(404)
    debug('get_neighbours: {} for {}'.format(len(result), id_))
    return jsonify(result)
