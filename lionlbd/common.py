from __future__ import print_function
from __future__ import absolute_import

import re

from flask import request

from logging import debug

from lionlbd.config import FILTER_TYPE_PARAMETER, YEAR_PARAMETER


def get_filter_type():
    filter_type_terms = [
        t for t in request.args.getlist(FILTER_TYPE_PARAMETER)
        if t and not t.isspace()
    ]
    debug('filter type terms: {}'.format(filter_type_terms))
    if len(filter_type_terms) == 0:
        return None
    else:
        return filter_type_terms


def get_year():
    return _get_int_argument(YEAR_PARAMETER, default=None)


def _get_int_argument(name, default=0, minimum=None, maximum=None):
    """Return value of integer argument in request."""
    values = [
        int(v) for v in request.args.getlist(name)
        if _get_int_argument.re.match(v)    # ignore invalid
    ]
    # filter out-of-bounds values
    if minimum is not None:
        values = [v for v in values if v >= minimum]
    if maximum is not None:
        values = [v for v in values if v <= maximum]
    if not values:
        return default
    else:
        return values[0]    # pick first if multiple
_get_int_argument.re = re.compile(r'^[+-]?[0-9]+$')
