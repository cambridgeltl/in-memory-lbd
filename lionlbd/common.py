from __future__ import print_function
from __future__ import absolute_import

from flask import request

from logging import debug

from lionlbd.config import FILTER_TYPE_PARAMETER


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
