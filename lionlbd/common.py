from __future__ import print_function
from __future__ import absolute_import

import os
import re
import time

from flask import request

from functools import wraps
from collections import defaultdict
from logging import debug, info

from lionlbd.config import YEAR_PARAMETER, METRIC_PARAMETER
from lionlbd.config import FILTER_TYPE_PARAMETER
from lionlbd.config import LIMIT_PARAMETER, OFFSET_PARAMETER


def get_metric():
    values = [
        t for t in request.args.getlist(METRIC_PARAMETER)
        if t and not t.isspace()
    ]
    if not values:
        return None
    else:
        return values[0]


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


def get_limit():
    return _get_int_argument(LIMIT_PARAMETER, default=None)


def get_offset():
    return _get_int_argument(OFFSET_PARAMETER, default=0)


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


def _format_usage(usage, unit):
    """Helper for memory usage functions."""
    denominator = {
        'b': 1,
        'k': 1024.,
        'm': 1024.**2,
        'g': 1024.**3,
    }
    if unit is None:
        return usage
    else:
        denom = denominator[unit.lower()]
        return '{:.2f}{}'.format(usage/denom, unit)


def memory_usage(unit=None):
    import psutil
    process = psutil.Process(os.getpid())
    usage = process.memory_info().rss
    return _format_usage(usage, unit)


def timed(func, log=info):
    """Decorator for logging execution time."""
    @wraps(func)
    def wrapper(*args, **argv):
        start = time.time()
        value = func(*args, **argv)
        end = time.time()
        diff = end-start
        fname = _func_name(func, *args)
        timed._total[fname] += diff
        timed._count[fname] += 1
        fstr = _func_arg_str(func, *args, **argv)
        log('timed {}: {:.2f}s (total {:.2f}s for {})'.format(
            fstr, diff, timed._total[fname], timed._count[fname]))
        return value
    return wrapper
timed._total = defaultdict(float)
timed._count = defaultdict(int)


def _is_method(f):
    """Return True if f is a method, False otherwise."""
    from inspect import getargspec
    # https://stackoverflow.com/a/19315046
    try:
        spec = getargspec(f)
        return spec.args and spec.args[0] == 'self'
    except TypeError:    # fails for cython functions
        return False    # default


def _func_arg_str(func, *args, **argv):
    """Pretty output helper for decorators."""
    return '{}({})'.format(_func_name(func, *args),
                           _arg_str(func, *args, **argv))


def _func_name(func, *args):
    """Pretty output helper for decorators."""
    if not _is_method(func):
        return func.__name__
    else:
        return '{}.{}'.format(type(args[0]).__name__, func.__name__)


def _arg_str(func, *args, **argv):
    """Pretty output helper for decorators."""
    max_args, max_len = 1, 1
    if _is_method(func):
        args = args[1:]    # skip self
    abbr = args if len(args) <= max_args else args[:max_args] + ('...',)
    abbr = [a if not (isinstance(a, list) and len(a) > max_len)
            else str(a[:max_len]) + '...' for a in abbr]
    s = ', '.join([str(a) for a in abbr] +
                  ['{}={}'.format(k, v) for k, v in argv.items()])
    if len(s) > 80:
        s = s[:75] + ' ...'
    return s
