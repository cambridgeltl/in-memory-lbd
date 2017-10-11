#!/usr/bin/env python

# Functions for reading Neo4j CSV.

from __future__ import print_function
from __future__ import absolute_import

import csv

from array import array
from itertools import izip
from collections import namedtuple
from pydoc import locate
from logging import debug, info, warn, error

from lionlbd.common import timed


@timed
def load_nodes(fn):
    """Load nodes from Neo4j CSV, return namedtuples.

    Stores Neo4j types in namedtuple as _field_types list.
    """
    colnames, coltypes, data = load_csv(fn)

    # more pythonic names
    for old, new in (('ID', 'id'), ('LABEL', 'label')):
        _rename_column(colnames, old, new)

    # nodes have redundant "id" and "OID" fields; remove the latter
    try:
        oid_idx = colnames.index('OID')
    except ValueError:
        raise ValueError('expected column OID, got {}'.format(old, colnames))
    for d in data:
        del d[oid_idx]
    del colnames[oid_idx]
    del coltypes[oid_idx]

    class_ = namedtuple('Node', ' '.join(colnames))
    class_._field_types = coltypes
    return map(class_._make, data)


@timed
def load_edges(fn):
    """Load edges from Neo4j CSV, return namedtuples.

    Stores Neo4j types in namedtuple as _field_types list.
    """
    colnames, coltypes, data = load_csv(fn)

    # more pythonic names
    for old, new in (('START_ID', 'start'), ('END_ID', 'end'),
                     ('TYPE', 'type')):
        _rename_column(colnames, old, new)

    class_ = namedtuple('Edge', ' '.join(colnames))
    class_._field_types = coltypes
    return map(class_._make, data)


def _rename_column(colnames, old, new):
    try:
        idx = colnames.index(old)
    except ValueError:
        raise ValueError('expected column {}, got {}'.format(old, colnames))
    colnames[idx] = new


def load_csv(fn):
    """Load Neo4j CSV, return lists of column names and types and data."""
    with open(fn) as f:
        reader = csv.reader(f)
        header = next(reader)
        colnames, coltypes = _parse_csv_header(header)
        debug('parsed header from {}: {}'.format(
            fn, list(zip(colnames, coltypes))))
        parsers = [_get_parser(t) for t in coltypes]
        data = []
        for row in reader:
            parsed = []
            for parser, value in izip(parsers, row):
                parsed.append(parser(value))
            data.append(parsed)
        debug('loaded {} rows from {}'.format(len(data), fn))
        return colnames, coltypes, data


def transpose(namedtuples):
    """Switch rows and columns in list of namedtuples with field types.

    Requires _field_types static variable as assigned by load_nodes()
    and load_edges().
    """
    if not namedtuples:
        raise ValueError('no namedtuples')
    class_ = type(namedtuples[0])
    try:
        field_types = class_._field_types
    except:
        raise ValueError('expected namedtuple with _field_types, got {}'\
                         .format(namedtuples[0]))
    field_names = class_._fields
    transposed = []
    for i, type_ in enumerate(field_types):
        if type_ in (int, float):
            tc = array_type_code(type_)
            a = array(tc, [t[i] for t in namedtuples])
        else:
            a = [t[i] for t in namedtuples]
        transposed.append(a)
    return class_(*transposed)


def _parse_csv_header(header):
    """Parser Neo4j CSV header, return lists of column names and types."""
    colnames, coltypes = [], []
    for col in header:
        if col.startswith(':'):    # handle ":TYPE" etc.
            col = col[1:]
        if ':' in col:    # name:type
            name, type_ = col.split(':', 1)
        else:    # default type
            name, type_ = col, 'str'
        colnames.append(name)
        coltypes.append(type_)
    return colnames, coltypes


def _get_parser(type_):
    """Return function for parsing string to given Neo4j type."""
    if type_.endswith('[]'):
        type_ = locate(type_[:-2])
        tc = array_type_code(type_)
        return lambda v: array(tc, map(type_, v.split(';')))
    else:
        return locate(type_)


def array_type_code(type_):
    # see https://docs.python.org/2/library/array.html
    if type_ is int:
        return 'i'
    elif type_ is float:
        return 'f'
    else:
        raise NotImplementedError(type_)
