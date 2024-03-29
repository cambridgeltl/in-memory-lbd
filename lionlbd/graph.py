#!/usr/bin/env python

# In-memory Neo4j graph.

from __future__ import print_function
from __future__ import absolute_import

import pyximport; pyximport.install()
from lionlbd.cgraph import open_discovery_core, mark_type_filtered
from lionlbd.cgraph import reindex_float

from array import array
from itertools import izip
from bisect import bisect
from logging import debug, info, warn, error

from numpy import argsort

from lionlbd.config import METRIC_PREFIX, METRIC_SUFFIX
from lionlbd.common import timed
from lionlbd.neo4jcsv import transpose, array_type_code
from lionlbd.lbdinterface import LbdInterface, LbdFilters


class Graph(LbdInterface):
    """In-memory Neo4j graph."""

    Filters = LbdFilters

    def __init__(self, nodes, edges, metadata):
        """Initialize Graph.

        Note: sorts given lists of nodes and edges.
        """
        if not nodes:
            raise ValueError('no nodes')
        if not edges:
            raise ValueError('no edges')
        if not metadata:
            raise ValueError('no metadata')
        if len(metadata) != 1:
            raise ValueError('expected one meta item, got {}'.format(len(meta)))
        self._metadata = metadata[0]

        nodes, edges = self._to_directed_graph(nodes, edges)
        self._node_count = len(nodes)
        self._edge_count = len(edges)

        self._sort_nodes_and_edges(nodes, edges)

        self._node_idx_by_id = self._create_node_idx_mapping(nodes)

        self._min_year, self._max_year = edges[0].year, edges[-1].year
        info('min edge year {}, max edge year {}'.format(
            self._min_year, self._max_year))

        self._metrics = self._get_edge_metrics(edges[0])
        info('metrics: {}'.format(self.get_metric_information()))

        self._nodes_t = transpose(nodes)
        self._edges_t = transpose(edges)
        nodes, edges = None, None    # release

        self._node_type_map = self._create_binary_type_map(self._nodes_t)
        self._node_types = list(sorted(self._node_type_map.keys()))
        self._nodes_t = self._map_types(self._nodes_t, self._node_type_map)

        self._edges_t = self._ids_to_indices(
            self._edges_t, self._node_idx_by_id)

        # TODO lift this block into a separate function
        self._weights_by_metric_and_year = {}
        for m_idx, m_name, m_type in self._metrics:
            self._weights_by_metric_and_year[m_name] = self._group_by_year(
                m_name, m_type, self._edges_t[m_idx], self._edges_t.year,
                self._min_year, self._max_year)
            self._edges_t = self._remove_metric(
                m_name, self._edges_t, self._metrics)

        self._metric_ranges = self._find_metric_ranges(
            self._weights_by_metric_and_year)

        self._neighbour_idx = self._create_neighbour_sequences(
            self._node_count, self._edges_t)

        # TODO is this needed?
        self._edges_from = self._create_edge_sequences(
            self._node_count, self._edges_t)

        self._edge_from_to = self._create_edge_from_to(
            self._node_count, self._edges_t)

        self._weights_from_cache = {}    # lazy init
        # for year in range(self._min_year, self._max_year+1):
        #     self._get_weights_from('count', year)    # precache (TODO)
        self._get_weights_from('count', self._max_year)    # precache (TODO)

        debug('initialized Graph: {}'.format(self.stats_str()))

    @timed
    def neighbours(self, id_, metric, year=None, filters=None, limit=None,
                   offset=0, exclude_neighbours_of=None):
        """Return neighbours of node."""
        idx = self._get_node_idx(id_)

        metric, year, filters, limit, offset = self._validate_common_args(
            metric, year, filters, limit, offset)
        if exclude_neighbours_of == id_:
            # https://github.com/cambridgeltl/lion-lbd/issues/117
            error('neighbours(): exclude_neighbours_of == id ({})'.format(id_))
            exclude_neighbours_of = None    # ignore

        filter_node = self._get_node_filter(filters.b_types)
        filter_edge = self._get_weight_filter(filters.min_weight,
                                              filters.max_weight)
        if exclude_neighbours_of is None:
            excluded_indices = set()    # exclude nothing
        else:
            x_idx = self._get_node_idx(exclude_neighbours_of)
            excluded_indices = set(self._neighbour_indices(x_idx, metric, year))

        scores, n_indices = [], []
        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)
        for n_idx, e_weight in izip(neighbour_idx[idx], weights_from[idx]):
            if (filter_node(n_idx) or filter_edge(e_weight) or
                n_idx in excluded_indices):
                continue
            scores.append(e_weight)
            n_indices.append(n_idx)

        argsorted = argsort(scores)[::-1]    # TODO: argpartition if limit?
        end_idx = offset+limit if limit is not None else len(scores)
        end_idx = min(end_idx, len(scores))

        node_ids, node_scores = [], []
        node_id = self._nodes_t.id
        for idx in argsorted[offset:end_idx]:
            node_ids.append(node_id[n_indices[idx]])
            node_scores.append(scores[idx])
        score_type = self._metric_type(metric)
        node_scores = [score_type(s) for s in node_scores]
        return node_ids, node_scores

    @timed
    def _neighbour_indices(self, idx, metric, year):
        """Internal partial implementation of neighbours()."""
        # Note: weights_from is here used to implement the implicit
        # year constraint. The actual metric values are not required.
        weights_from = self._get_weights_from(metric, year)
        neighbour_count = len(weights_from[idx])
        # TODO: consider islice to avoid creating new list?
        return self._neighbour_idx[idx][:neighbour_count]

    @timed
    def closed_discovery(self, a_id, c_id, metric, agg_func, year=None,
                         filters=None, limit=None, offset=0, exists_only=False):
        a_idx, c_idx = self._get_node_idx(a_id), self._get_node_idx(c_id)

        metric, year, filters, limit, offset = self._validate_common_args(
            metric, year, filters, limit, offset)

        agg = self._get_agg_function(agg_func)

        filter_node = self._get_node_filter(filters.b_types)
        filter_edge = self._get_weight_filter(filters.min_weight,
                                              filters.max_weight)

        weights_from = self._get_weights_from(metric, year)

        # Note: the following assumes that edges are symmetric.
        # Swap "A" and "C" if the former has more neighbours than the
        # latter (fewer iterations).
        if len(weights_from[a_idx]) > len(weights_from[c_idx]):
            debug('closed_discovery: swap to A: {}, C: {}'.format(c_id, a_id))
            a_idx, c_idx = c_idx, a_idx

        scores, b_indices = [], []
        neighbour_idx = self._neighbour_idx
        edge_from_to = self._edge_from_to
        edge_weight = self._weights_by_metric_and_year[metric][year]
        for b_idx, e1_weight in izip(neighbour_idx[a_idx], weights_from[a_idx]):
            if filter_node(b_idx) or filter_edge(e1_weight):
                continue
            e2_idx = edge_from_to[b_idx].get(c_idx)
            if e2_idx is None:
                continue    # no B-C edge
            if e2_idx >= len(edge_weight):
                continue    # implicit year filter
            e2_weight = edge_weight[e2_idx]
            if filter_edge(e2_weight):
                continue
            if exists_only:
                return True
            scores.append(agg(e1_weight, e2_weight))
            b_indices.append(b_idx)

        if exists_only:
            return False

        argsorted = argsort(scores)[::-1]    # TODO: argpartition if limit?
        end_idx = offset+limit if limit is not None else len(scores)
        end_idx = min(end_idx, len(scores))

        node_ids, node_scores = [], []
        node_id = self._nodes_t.id
        for idx in argsorted[offset:end_idx]:
            node_ids.append(node_id[b_indices[idx]])
            node_scores.append(scores[idx])
        return node_ids, node_scores

    @timed
    def open_discovery(self, a_id, metric, agg_func, acc_func, year=None,
                       filters=None, limit=None, offset=0):
        """Get 2nd-degree neighbours of node.

        Excludes the starting node and its 1st-degree neighbours.
        """
        a_idx = self._get_node_idx(a_id)

        metric, year, filters, limit, offset = self._validate_common_args(
            metric, year, filters, limit, offset)

        agg_func = self._validate_aggregation_function(agg_func)
        acc_func = self._validate_accumulation_function(acc_func)

        node_count = self._node_count

        # Flag nodes to exclude for fast access in inner loop.
        exclude_idx = array('b', [0]) * node_count
        # TODO: include constraints other than year?
        b_indices = self._neighbour_indices(a_idx, metric, year)
        for b_idx in b_indices:
            exclude_idx[b_idx] = 1
        exclude_idx[a_idx] = 1

        mark_type_filtered(exclude_idx, self._nodes_t.type, filters.c_types,
                           self._node_type_map)

        # accumulate scores by node in array
        score = array('f', [0]) * node_count
        is_c_idx = array('b', [0]) * node_count

        neighbour_idx = self._neighbour_idx
        weights_from = self._get_weights_from(metric, year)

        filter_b_node = self._get_node_filter(filters.b_types)
        filter_edge = self._get_weight_filter(filters.min_weight,
                                              filters.max_weight)

        open_discovery_core(a_idx, neighbour_idx, weights_from, score,
                            is_c_idx, exclude_idx, filter_b_node,
                            filter_edge, agg_func, acc_func)

        argsorted = argsort(score)[::-1]    # TODO: argpartition if limit?
        limit = limit if limit is not None else node_count
        end_idx = offset+limit if limit is not None else len(scores)

        node_ids, node_scores, result_idx = [], [], 0
        node_id = self._nodes_t.id
        for idx in argsorted:
            if len(node_ids) >= limit:
                break
            if not is_c_idx[idx]:
                continue
            if result_idx >= offset:
                node_ids.append(node_id[idx])
                node_scores.append(score[idx])
            result_idx += 1
        return node_ids, node_scores

    def _get_metric_data_filler(self, metrics, year, history):
        """Return function filling in metric values for edge."""
        # Helper for subgraph_edges()
        mtype, weight = self._metric_type, self._weights_by_metric_and_year
        if not history:
            metric_data = [(m, mtype(m), weight[m][year]) for m in metrics]
            def fill_metric_values(edge, idx):
                for metric, type_, edge_weight in metric_data:
                    edge[metric] = type_(edge_weight[idx])
            return fill_metric_values
        else:
            metric_data = [(m, mtype(m), weight[m]) for m in metrics]
            edge_year = self._edges_t.year
            def fill_metric_values(edge, idx):
                for metric, type_, edge_weight in metric_data:
                    edge[metric] = [type_(edge_weight[y][idx])
                                    for y in xrange(edge_year[idx], year+1)]
            return fill_metric_values

    def subgraph_edges(self, nodes, metrics, year=None, filters=None,
                       exclude=None, history=False):
        if not metrics:
            raise ValueError('no metrics')
        if not isinstance(nodes, list):
            nodes = list(nodes)
        node_indices = [self._get_node_idx(i) for i in nodes]
        metrics = self._validate_metrics(metrics)
        year = self._validate_year(year)
        filters = self._validate_filters(filters)
        exclude = [] if exclude is None else exclude
        exclude_indices = set([self._get_node_idx(i) for i in exclude])

        filter_edge = self._get_weight_filter(filters.min_weight,
                                              filters.max_weight)
        filtered_weight = self._weights_by_metric_and_year[metrics[0]][year]
        fill_metric_values = self._get_metric_data_filler(metrics, year,
                                                          history)

        edges = []
        edge_year = self._edges_t.year
        edge_type = self._edges_t.type
        node_id_idx = zip(nodes, node_indices)
        for i, (n1_id, n1_idx) in enumerate(node_id_idx):
            for j in xrange(i+1, len(node_indices)):
                n2_id, n2_idx = node_id_idx[j]
                if n1_idx in exclude_indices and n2_idx in exclude_indices:
                    continue    # skip (possible) edge between "exclude" nodes
                edge_idx = self._edge_from_to[n1_idx].get(n2_idx)
                if edge_idx is None:
                    continue    # no edge there
                if edge_year[edge_idx] > year:
                    continue    # edge only appears after given year
                if filter_edge(filtered_weight[edge_idx]):
                    continue
                edge = {
                    'start': n1_id,
                    'end': n2_id,
                    'type': edge_type[edge_idx],
                    'year': edge_year[edge_idx],
                }
                fill_metric_values(edge, edge_idx)
                edges.append(edge)

        return edges

    def discoverable_edges(self, after, until=None):
        # validate after and until parameters
        min_year, max_year = self.get_year_range()
        if after <= min_year:
            raise ValueError('after ({}) <= min ({})'.format(after, min_year))
        if after >= max_year:
            raise ValueError('after ({}) >= max ({})'.format(after, max_year))
        if until is not None and until <= after:
            raise ValueError('until ({}) <= after ({})'.format(until, after))
        if until is not None and until > max_year:
            raise ValueError('until ({}) >= max ({})'.format(until, max_year))
        if until is None:
            until = max_year

        # TODO make use of edge sort by year (bisect)
        node_id = self._nodes_t.id
        edge_start, edge_end = self._edges_t.start, self._edges_t.end
        discoverable = []
        for edge_idx, edge_year in enumerate(self._edges_t.year):
            if edge_year <= after:
                continue
            elif until is not None and edge_year > until:
                continue
            start_idx, end_idx = edge_start[edge_idx], edge_end[edge_idx]
            start_id, end_id = node_id[start_idx], node_id[end_idx]

            path_exists = self.closed_discovery(
                start_id, end_id, metric=None, agg_func=None, year=after,
                exists_only=True)
            if not path_exists:
                continue

            discoverable.append({
                'start': start_id,
                'end': end_id,
                'year': edge_year
            })
        return discoverable

    def get_nodes(self, ids, metric=None, year=None, filters=None,
                  exclude_neighbours_of=None, history=False):
        indices = [self._get_node_idx(i) for i in ids]
        metric = self._validate_metric(metric)
        year = self._validate_year(year)
        filters = self._validate_filters(filters)

        if not history:
            get_counts = lambda seq, idx: seq[idx]
        else:
            get_counts = lambda seq, idx: list(seq[:idx])

        nodes = []
        node_id = self._nodes_t.id
        node_type = self._nodes_t.type
        node_text = self._nodes_t.text
        node_year = self._nodes_t.year
        node_count = self._nodes_t.count
        node_doc_count = self._nodes_t.doc_count
        inv_type_map = { v: k for k, v in self._node_type_map.items() }
        for i in indices:
            i_year = node_year[i]
            if i_year > year:
                raise ValueError('get_nodes: node_year > year')    # TODO
            year_idx = year - i_year
            id_ = node_id[i]
            # TODO: only need neighbour count, avoid creating list and
            # avoid metric
            neighbours, _ = self.neighbours(
                id_, metric=metric, year=year, filters=filters,
                exclude_neighbours_of=exclude_neighbours_of)
            nodes.append({
                'id': id_,
                'type': inv_type_map[node_type[i]],
                'text': node_text[i],
                'year': i_year,
                'count': get_counts(node_count[i], year_idx),
                'doc_count': get_counts(node_doc_count[i], year_idx),
                'edge_count': len(neighbours),
            })
        return nodes

    def get_year_range(self):
        return self._min_year, self._max_year

    def get_types(self):
        return self._node_types

    def get_metrics(self):
        """Return edge weight metrics used in the graph.

        Returns:
            list of str: edge metric names.
        """
        return [name for index, name, type_ in self._metrics]

    def get_metric_range(self, metric):
        self._validate_metric(metric)
        return self._metric_ranges[metric]

    def get_aggregation_functions(self):
        return ['min', 'avg', 'max']    # TODO

    def get_accumulation_functions(self):
        return ['sum', 'max']    # TODO

    def meta_information(self):
        # convert into JSON-serializable form
        metadata = {
            k: v if not isinstance(v, array) else list(v)
            for k, v in self._metadata._asdict().iteritems()
        }
        return metadata

    def _get_node_idx(self, id_):
        try:
            idx = self._node_idx_by_id[id_]
        except KeyError:
            raise KeyError('unknown node id: {}'.format(id_))
        return idx

    def _metric_type(self, metric):
        """Return type of values for metric (e.g. float)."""
        for index, name, type_ in self._metrics:
            if name == metric:
                return type_
        raise ValueError(metric)

    def _get_node_filter(self, types=None):
        """Return function determining whether to filter out a node."""
        if types is None:
            return lambda n_idx: False
        else:
            type_bits = [self._node_type_map[t] for t in types]
            type_mask = reduce(lambda x,y: x|y, type_bits)
            # debug('type_mask {} for {}'.format(type_mask, types))
            node_types = self._nodes_t.type
            return lambda n_idx: not (node_types[n_idx] & type_mask)

    def _get_weight_filter(self, min_weight=None, max_weight=None):
        """Return function determining whether to filter out an edge."""
        if min_weight is None and max_weight is None:
            return lambda w: False
        elif min_weight is not None and max_weight is None:
            return lambda w: w < min_weight
        elif min_weight is None and max_weight is not None:
            return lambda w: w > max_weight
        else:
            assert min_weight is not None and max_weight is not None
            return lambda w: w < min_weight or w > max_weight

    @timed
    def _calculate_weights_from(self, metric, year):
        info('calculating weights_from for metric {}, year {} ...'.format(
            metric, year))
        edge_weight = self._weights_by_metric_and_year[metric][year]
        type_code = array_type_code(self._metric_type(metric))
        idx_after = bisect(self._edges_t.year, year)    # sorted by year
        return reindex_float(self._node_count, self._edges_t.start,
                             edge_weight, idx_after)

    def _get_weights_from(self, metric, year):
        if metric not in self._weights_from_cache:
            self._weights_from_cache[metric] = {}
        if year not in self._weights_from_cache[metric]:
            self._weights_from_cache[metric][year] \
                = self._calculate_weights_from(metric, year)
        return self._weights_from_cache[metric][year]

    def _validate_common_args(self, metric, year, filters, limit=None,
                              offset=0):
        """Validate common arguments, applying defaults when applicable."""
        return (
            self._validate_metric(metric),
            self._validate_year(year),
            self._validate_filters(filters),
            self._validate_limit(limit),
            self._validate_offset(offset)
        )

    def stats_str(self):
        """Return Graph statistics as string."""
        return '{} nodes, {} edges'.format(self._node_count, self._edge_count)

    @staticmethod
    def _sort_nodes_and_edges(nodes, edges):
        """Sort node and edge arrays for efficient storage and search."""

        # sort edges by year
        edges.sort(key=lambda e: e.year)

        # for each node, identify earliest year with an edge
        first_edge_year = {}
        for edge in edges:
            for n_id in (edge.start, edge.end):
                if n_id not in first_edge_year:
                    first_edge_year[n_id] = edge.year
        max_year = edges[-1].year
        edgeless = [n for n in nodes if n.id not in first_edge_year]
        if edgeless:
            ids = [n.id for n in edgeless]
            id_str = (', '.join(ids) if len(ids) < 5 else
                      ', '.join(ids[:5]) + '...')
            warn('{} nodes without edges ({})'.format(len(edgeless), ids))
        for node in edgeless:
            # sort in after other nodes, otherwise arbitrary order
            first_edge_year[node.id] = max_year + 1

        # sort nodes by year of earliest edge
        nodes.sort(key=lambda n: first_edge_year[n.id])

    @staticmethod
    def _create_node_idx_mapping(nodes):
        """Create mapping from node ID to index in given sequence."""
        node_idx_by_id = {}
        for idx, node in enumerate(nodes):
            if node.id in node_idx_by_id:
                raise ValueError('duplicate node ID {}'.format(node.id))
            node_idx_by_id[node.id] = idx
        return node_idx_by_id

    @staticmethod
    def _create_binary_type_map(nodes_t):
        """Create binary encoding of node types.

        Note:
            Expects nodes_t to be result of transpose(nodes).
        """
        types = sorted(list(set(nodes_t.type)))
        info('types: {}'.format(types))
        type_map = { t: 1<<i for i, t in enumerate(types) }
        debug('binary type mapping: {}'.format(type_map))
        return type_map

    @staticmethod
    def _map_types(nodes_t, type_map):
        """Replace node types with values from given map.

        Note:
            Expects nodes_t to be result of transpose(nodes).
        """
        nodes_d = nodes_t._asdict()
        nodes_d['type'] = array('i', (type_map[t] for t in nodes_t.type))
        class_ = type(nodes_t)
        return class_(**nodes_d)

    @staticmethod
    def _ids_to_indices(edges_t, node_idx_by_id):
        """Replace node ids with indices in edge data.

        Note:
            Expects edges_t to be result of transpose(edges).
        """
        edges_d = edges_t._asdict()
        edges_d['start'] = array('i',(node_idx_by_id[i] for i in edges_t.start))
        edges_d['end'] = array('i', (node_idx_by_id[i] for i in edges_t.end))
        class_ = type(edges_t)
        return class_(**edges_d)

    @staticmethod
    @timed
    def _find_metric_ranges(weights_by_metric_and_year):
        ranges = {}
        for metric, weights_by_year in weights_by_metric_and_year.iteritems():
            m_min, m_max = None, None
            for weights in weights_by_year.values():
                y_min, y_max = min(weights), max(weights)
                m_min = y_min if m_min is None or y_min < m_min else m_min
                m_max = y_max if m_max is None or y_max > m_max else m_max
            ranges[metric] = (m_min, m_max)
            debug('range for {}: [{}, {}]'.format(metric, m_min, m_max))
        return ranges

    @staticmethod
    @timed
    def _group_by_year(name, m_type, edge_weights, edge_years, min_year,
                       max_year):
        """Group metric values by year.

        Note:
            Expects edges_t to be result of transpose(edges) and sorted by year.
        """
        type_code = array_type_code(m_type)
        if type_code != 'f':
            warn('converting weight type {} to float'.format(type_code))
            type_code = 'f'
        weights_by_year = {}
        for year in range(min_year, max_year+1):
            y_idx = year - max_year - 1
            idx_limit = bisect(edge_years, year)    # edges sorted by year
            weights_by_year[year] = array(
                type_code, (edge_weights[i][y_idx] for i in range(idx_limit)))
        return weights_by_year

    @staticmethod
    def _remove_metric(name, edges_t, metrics):
        """Remove metric from transposed edges.

        Use with _group_by_year(), avoids storing metrics redundantly.
        """
        edges_l = list(edges_t)
        for m_idx, m_name, m_type in metrics:
            if m_name == name:
                edges_l[m_idx] = None
        class_ = type(edges_t)
        return class_(*edges_l)

    @staticmethod
    @timed
    def _create_neighbour_sequences(node_count, edges_t):
        """Create index-based mapping from node to neighbouring nodes.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        neighbour_idx = [array('i') for _ in xrange(node_count)]
        for start, end in izip(edges_t.start, edges_t.end):
            neighbour_idx[start].append(end)
        return neighbour_idx

    @staticmethod
    def _create_edge_sequences(node_count, edges_t):
        """Create index-based mapping from node to edges.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        edges_from = [array('i') for _ in xrange(node_count)]
        for idx, start in enumerate(edges_t.start):
            edges_from[start].append(idx)
        return edges_from

    @staticmethod
    @timed
    def _create_edge_from_to(node_count, edges_t):
        """Create node index-based mapping m where m[f][t] is edge index.

        Note:
            Expects edges_t to be result of _ids_to_indices(transpose(edges)).
        """
        edge_from_to = [{} for _ in xrange(node_count)]
        for i, (start, end) in enumerate(izip(edges_t.start, edges_t.end)):
            if end in edge_from_to[start]:
                raise ValueError('duplicate edge {} - {}'.format(start, end))
            edge_from_to[start][end] = i
        return edge_from_to

    @staticmethod
    def _to_directed_graph(nodes, edges):
        """Given an undirected graph, return a directed equivalent.

        Note: produces shallow copies of edges, leading to referenced
        objects beings shared between edges.
        """
        if not edges:
            return nodes, edges

        # assume edges are namedtuples and all edges share the index
        # positions of the start and end values
        fields = type(edges[0])._fields
        START, END = fields.index('start'), fields.index('end')

        directed_edges = edges[:]
        for edge in edges:
            e_class = type(edge)
            e_list = list(edge)
            e_list[START], e_list[END] = e_list[END], e_list[START]
            directed_edges.append(e_class(*e_list))
        return nodes, directed_edges

    @staticmethod
    def _get_edge_metrics(edge):
        """Return list of metric (index, name, type) tuples for namedtuple."""
        metrics = []
        for index, field in enumerate(type(edge)._fields):
            if (field.startswith(METRIC_PREFIX) and
                field.endswith(METRIC_SUFFIX)):
                value = edge[index]
                if METRIC_SUFFIX:
                    name = field[len(METRIC_PREFIX):-len(METRIC_SUFFIX)]
                else:
                    name = field[len(METRIC_PREFIX):]
                try:
                    type_ = type(value[0])    # assume sequence
                except:
                    error('expected sequence of values, got {}'.format(
                        type(value)))
                    raise
                metrics.append((index, name, type_))
        if not metrics:
            warn('no metrics found for {}'.format(edge))
        else:
            debug('metrics for {}({}): {}'.format(
                type(edge).__name__, type(edge)._fields, metrics))
        return metrics

    @staticmethod
    def _get_agg_function(name):
        if name == 'min':
            return lambda a, b: min(a, b)    # TODO bad idea, remove
        elif name == 'sum':
            return lambda a, b: a + b
        elif name == 'avg':
            return lambda a, b: (a + b) / 2.0
        elif name == 'max':
            return lambda a, b: max(a, b)
        elif name is None:
            return lambda a, b: None    # no aggregation / result unused
        else:
            raise NotImplementedError(name)

    @staticmethod
    def _get_acc_function(name):
        if name == 'max':
            return lambda a, b: max(a, b)
        if name == 'sum':
            return lambda a, b: a + b
        else:
            raise NotImplementedError(name)
