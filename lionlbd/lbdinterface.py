#!/usr/bin/env python

# Abstract base class for LBD support classes.

from __future__ import print_function
from __future__ import absolute_import

from six import string_types

from abc import ABCMeta, abstractmethod


class LbdFilters(object):
    """Filters applying to LBD queries."""

    def __init__(self, b_types=None, c_types=None, min_weight=None,
                 max_weight=None):
        """Initialize LbdFilters.

        Args:
            b_types: Sequence of type names (str) to restrict neighbours
                to, or None to allow any type.
            c_types: Sequence of type names (str) to restrict 2nd-degree
                neighbours to, or None to allow any type.
            min_weight (float): Minimum metric value to restrict to, or
                None for no lower bound.
            max_weight (float): Maximum metric value to restrict to, or
                None for no upper bound.

        Raises:
            ValueError if any argument has an inappropriate value.
        """
        try:
            b_types = self._validate_types(b_types)
        except:
            raise ValueError('LbdFilters b_types: {}'.format(b_types))

        try:
            c_types = self._validate_types(c_types)
        except:
            raise ValueError('LbdFilters c_types: {}'.format(c_types))

        if min_weight is not None and not isinstance(min_weight, (float, int)):
            raise ValueError('LbdFilters min_weight: {}'.format(min_weight))

        if max_weight is not None and not isinstance(max_weight, (float, int)):
            raise ValueError('LbdFilters max_weight: {}'.format(max_weight))

        self.b_types = b_types
        self.c_types = c_types
        self.min_weight = min_weight
        self.max_weight = max_weight

    @staticmethod
    def _validate_types(types):
        if types is None:
            return types
        else:
            if isinstance(types, string_types):
                raise ValueError()
            if any(n for n in types if not isinstance(n, string_types)):
                raise ValueError()
            return list(types)


class LbdInterface(object):
    __metaclass__ = ABCMeta

    """Abstract base class for LBD support classes."""

    @abstractmethod
    def neighbours(self, id_, metric, year=None, filters=None, limit=None,
                   offset=0):
        """Return neighbours of node with given ID.

        Ranks neighbours by the values of the given metric on the
        connecting edges.

        Args:
            id_ (str): ID of node whose neighbours to return.
            metric (str): Name of metric to use for ranking. Must be one
                of the values returned by get_metrics().
            year (int): The "current" year for purposes of the search, or
                None for the most recent year. Must be within the range
                returned by get_year_range().
            filters (LbdFilters): The filters to apply, or None for no
                filtering.
            limit (int): Maximum number of items to return, or None for
                no limit.
            offset (int): Offset of first item in ranked list to return.
                Used together with limit for paging.

        Raises:
            ValueError if any argument has an inappropriate value.

        Returns:
            Sequence of (ID, score) pairs.
        """

    @abstractmethod
    def closed_discovery(self, a_id, c_id, metric, agg_func, year=None,
                         filters=None, limit=None, offset=0):
        """Return common neighbours of nodes with given IDs.

        Ranks common neighbours by the values of the given metric,
        aggregated by the specified function into a score for the
        path.

        Args:
            a_id (str): ID of the "start" node for the search.
            c_id (str): ID of the "end" node for the search.
            metric (str): Name of metric to use for ranking. Must be one
                of the values returned by get_metrics().
            agg_func (str): Name of the function to use to combine metric
                values over a path. Must be one of the values returned
                by get_aggregation_functions().
            year (int): The "current" year for purposes of the search, or
                None for the most recent year. Must be within the range
                returned by get_year_range().
            filters (LbdFilters): The filters to apply, or None for no
                filtering.
            limit (int): Maximum number of items to return, or None for
                no limit.
            offset (int): Offset of first item in ranked list to return.
                Used together with limit for paging.

        Raises:
            ValueError if any argument has an inappropriate value.

        Returns:
            Sequence of (ID, score) pairs.

        """

    @abstractmethod
    def open_discovery(self, a_id, metric, agg_func, acc_func, year=None,
                       filters=None, limit=None, offset=0):
        """Return [TODO]

        Args:
            a_id (str): ID of [TODO]
            metric (str): Name of metric to use for ranking. Must be one
                of the values returned by get_metrics().
            agg_func (str): Name of the function to use to combine metric
                values over a path. Must be one of the values returned
                by get_aggregation_functions().
            acc_func (str): Name of the function to use to accumulate
                aggregated scores for multiple paths arriving at the same
                node. Must be one of the values returned by
                get_accumulation_functions().
            year (int): The "current" year for purposes of the search, or
                None for the most recent year. Must be within the range
                returned by get_year_range()
            filters (LbdFilters): The filters to apply, or None for no
                filtering.
            limit (int): Maximum number of items to return, or None for
                no limit.
            offset (int): Offset of first item in ranked list to return.
                Used together with limit for paging.

        Raises:
            ValueError if any argument has an inappropriate value.

        Returns:
            Sequence of (ID, score) pairs.
        """

    @abstractmethod
    def get_year_range(self):
        """Return minimum and maximum years in the graph.

        Returns:
            Pair of integers (min_year, max_year).
        """

    @abstractmethod
    def get_metric_range(self, metric):
        """Return minimum and maximum values for given metric.

        Args:
            metric (str): Name of metric to return range for. Must be
                one of the values returned by get_metrics().

        Returns:
            Pair of floats (min_value, max_value).
        """

    @abstractmethod
    def get_types(self):
        """Return node types used in the graph.

        Returns:
            list of str: node types.
        """

    @abstractmethod
    def get_metrics(self):
        """Return edge weight metrics used in the graph.

        Returns:
            list of str: edge metric names.
        """

    @abstractmethod
    def get_aggregation_functions(self):
        """Return list of supported aggregation function names.

        Aggregation functions combine the edge weights along a path
        into a score for node ranking in closed_discovery() and
        open_discovery().

        Returns:
            list of str: aggregation function names.
        """

    @abstractmethod
    def get_accumulation_functions(self):
        """Return list of supported accumulation function names.

        Accumulation functions total the aggregated scores along multiple
        paths to a node in open_discovery().

        Returns:
            list of str: accumulation function names.
        """

    @abstractmethod
    def subgraph(self, nodes, metric, year=None, filters=None):
        """Return subgraph containing edges between given nodes.

        Args:
            nodes (list of str): IDs of the nodes of the subgraph.
            metric (str): Name of metric to use. Must be one
                of the values returned by get_metrics().
            year (int): The "current" year, or None for the most recent year.
                Edges only appearing after the given year are excluded from
                the subgraph.
            filters (LbdFilters): The filters to apply, or None for no
                filtering.
        """

    @abstractmethod
    def get_node(self):
        """[TODO]"""

    @abstractmethod
    def meta_information(self):
        """[TODO]"""
