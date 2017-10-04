from __future__ import print_function
from __future__ import absolute_import

import logging

from flask import abort, jsonify
from logging import debug, info, warn, error

from lionlbd import app
from lionlbd import graph


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()


@app.route('/neighbours/<id_>')
def get_neighbours(id_):
    try:
        neighbours = graph.get_neighbours(id_)
    except KeyError, e:
        warn(e)
        abort(404)
    debug('get_neighbours: {} for {}'.format(len(neighbours), id_))
    return jsonify([n._asdict() for n in neighbours])
