from __future__ import print_function
from __future__ import absolute_import

import json
import logging

from flask import request, abort, jsonify
from logging import debug, info, warn, error, exception

from lionlbd import app
from lionlbd import graph
from lionlbd.common import get_metric, get_year, get_filter_type
from lionlbd.common import get_limit, get_offset, get_until


@app.route('/')
@app.route('/index')
def index():
    return graph.stats_str()


# def ujsonify(obj):
#     from flask.globals import current_app
#     from ujson import dumps
#     return current_app.response_class((dumps(obj, indent=2), '\n'),
#                                       mimetype='application/json')


@app.route('/graph/<method>')
def invoke_method(method):
    """Invoke named method on graph."""
    # get method on graph
    try:
        name, method = method, getattr(graph, method)
    except AttributeError:
        error('invoke_method(): no method "{}"'.format(method))
        abort(404)

    # get caller parameters
    params = request.args.get('params')
    try:
        params = json.loads(params)
    except ValueError:
        exception('invoke_method(): failed to loads params "{}"'.format(params))
        abort(400)
    debug('invoke_method() params: {}'.format(params))

    # convert any non-None "filters" parameter to LbdFilters (special case)
    if params.get('filters', None) is not None:
        try:
            params['filters'] = graph.Filters(**params['filters'])
        except:
            exception('invoke_method(): failed to create filters "{}"'.format(
                params['filters']))
            abort(400)

    # invoke the method
    try:
        result = method(**params)
    except:
        exception('invoke_method(): failed to invoke {}({})'.format(
            name, params))
        abort(400)

    debug('invoke_method(): {} returned {}'.format(name, str(result)[:40]))
    return jsonify(result)


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


@app.route('/discoverable_edges/<after>')
def get_discoverable_edges(after):
    try:
        after = int(after)
    except ValueError:
        abort(400)
    until = get_until()
    result = graph.discoverable_edges(after, until)
    return jsonify(result)
