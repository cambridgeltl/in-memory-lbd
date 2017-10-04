from __future__ import print_function
from __future__ import absolute_import

import logging

from flask import abort, jsonify
from logging import debug, info, warn, error

from lionlbd import app
from lionlbd import graph
from lionlbd.common import get_filter_type


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()


@app.route('/neighbours/<id_>')
def get_neighbours(id_):
    types = get_filter_type()
    try:
        neighbours = graph.get_neighbours(id_, types=types)
    except KeyError, e:
        warn(e)
        abort(404)
    debug('get_neighbours: {} for {}'.format(len(neighbours), id_))
    return jsonify([n._asdict() for n in neighbours])
