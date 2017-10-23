from __future__ import print_function
from __future__ import absolute_import

import logging

from flask import abort, jsonify
from logging import debug, info, warn, error

from lionlbd import app
from lionlbd import graph
from lionlbd.common import get_metric, get_year, get_filter_type
from lionlbd.common import get_limit, get_offset


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()


# def ujsonify(obj):
#     from flask.globals import current_app
#     from ujson import dumps
#     return current_app.response_class((dumps(obj, indent=2), '\n'),
#                                       mimetype='application/json')


@app.route('/extend_graph/<id_>')
def extend_graph(id_):
    metric = get_metric()
    year = get_year()
    types = get_filter_type()
    filters = graph.Filters(types)
    limit = get_limit()
    offset = get_offset()

    # TODO existing nodes

    try:
        neighbours = graph.neighbours(
            id_,
            metric=metric,
            year=year,
            filters=filters,
            limit=limit,
            offset=offset
        )
    except KeyError, e:
        warn(e)
        abort(404)

    node_ids = [id_] + [n['B'] for n in neighbours]
    edges = graph.subgraph(
        node_ids,
        ['count'],
        year=year,
        filters=filters
    )

    return jsonify([neighbours, edges])

    
@app.route('/neighbours/<id_>')
def get_neighbours(id_):
    metric = get_metric()
    year = get_year()
    types = get_filter_type()
    filters = graph.Filters(types, types)
    limit = get_limit()
    offset = get_offset()
    try:
        result = graph.neighbours(
            id_,
            metric=metric,
            year=year,
            filters=filters,
            limit=limit,
            offset=offset
        )
    except KeyError, e:
        warn(e)
        abort(404)
    debug('get_neighbours: {} for {}'.format(len(result), id_))
    return jsonify(result)


@app.route('/neighbours2/<id_>')
def get_2nd_neighbours(id_):
    metric = get_metric()
    year = get_year()
    types = get_filter_type()
    filters = graph.Filters(types, types)
    limit = get_limit()
    offset = get_offset()
    try:
        result = graph.open_discovery(
            id_,
            metric=metric,
            agg_func='avg',    # TODO
            acc_func='max',    # TODO
            year=year,
            filters=filters,
            limit=limit,
            offset=offset
        )
    except KeyError, e:
        warn(e)
        abort(404)
    debug('get_2nd_neighbours: {} for {}'.format(len(result), id_))
    return jsonify(result)


@app.route('/closed_discovery/<a_id>/to/<c_id>')
def closed_discovery(a_id, c_id):
    metric = get_metric()
    year = get_year()
    types = get_filter_type()
    filters = graph.Filters(types, types)
    limit = get_limit()
    offset = get_offset()
    try:
        result = graph.closed_discovery(
            a_id, c_id,
            metric=metric,
            agg_func='avg',    # TODO
            year=year,
            filters=filters,
            limit=limit,
            offset=offset
        )
    except KeyError, e:
        warn(e)
        abort(404)
    return jsonify(result)
