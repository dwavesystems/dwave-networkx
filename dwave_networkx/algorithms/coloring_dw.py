"""
TODO
"""

from __future__ import division, absolute_import

import sys
import math
import itertools

import dwave_networkx as dnx
from dwave_networkx.utils_dw.decorators import discrete_model_sampler

__all__ = ["min_vertex_coloring_dm", "is_vertex_coloring"]

# compatibility for python 2/3
if sys.version_info[0] == 2:
    range = xrange
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


@discrete_model_sampler(1)
def min_vertex_coloring_dm(G, sampler=None, **sampler_args):
    """TODO

    https://en.wikipedia.org/wiki/Brooks%27_theorem
    """

    # if the given graph is not connected, apply the function to each connected component
    # seperately.
    if not dnx.is_connected(G):
        coloring = {}
        for subG in dnx.connected_component_subgraphs(G):
            sub_coloring = min_vertex_coloring_dm(subG, sampler, **sampler_args)
            coloring.update(sub_coloring)
        return coloring

    n_nodes = len(G)  # number of nodes
    n_edges = len(G.edges())  # number of edges

    # ok, first up, we can eliminate a few graph types trivially

    # Graphs with no edges, have chromatic number 1
    if not n_edges:
        return {node: 0 for node in G}

    # Complete graphs have chromatic number N
    if n_edges == n_nodes * (n_nodes - 1) / 2:
        return {node: color for color, node in enumerate(G)}

    # The number of variables in the QUBO is approximately the number of nodes in the graph
    # times the number of potential colors, so we want as tight an upper bound on the
    # chromatic number (chi) as possible

    # we know that chi <= max degree, unless it is complete or a cycle graph of odd length,
    # in which case chi <= max degree + 1 (Brook's Theorem)
    # we have already taken care of complete graphs.
    if n_nodes % 2 == 1 and is_cycle(G):
        # odd cycle graphs need three colors
        chi_ub = 3
    else:
        chi_ub = max(G.degree(node) for node in G)

    # we also know that chi*(chi-1) <= 2*|E|, so using the quadratic formula
    chi_ub = min(chi_ub, _quadratic_chi_bound(n_edges))

    # now we can start coloring. Without loss of generality, we can determine some of
    # the node colors before trying to solve.
    partial_coloring, possible_colors, chi_lb = _partial_precolor(G, chi_ub)

    # ok, to get the rest of the coloring, we need to start building the QUBO. We do this
    # by assigning a variable x_v_c for each node v and color c. This variable will be 1
    # when node v is colored c, and 0 otherwise.

    # let's assign an index to each of the variables
    counter = itertools.count()
    x_vars = {v: {c: next(counter) for c in possible_colors[v]} for v in possible_colors}

    # now we have three different constraints we wish to add.

    # the first constraint enforces the coloring rule, that for each pair of vertices
    # u, v that share an edge, they should be different colors
    Q_neighbor = _vertex_different_colors_qubo(G, x_vars)

    # the second constraint enforces that each vertex has a single color assigned
    Q_vertex = _vertex_one_color_qubo(x_vars)

    # the third constraint is that we want a minimum vertex coloring, so we want to
    # disincentivize the colors we might not need.
    Q_min_color = _minimum_coloring_qubo(x_vars, chi_lb, chi_ub, magnitude=.75)

    # combine all three constraints and solve
    Q = Q_neighbor
    for (u, v), bias in iteritems(Q_vertex):
        if (u, v) in Q:
            Q[(u, v)] += bias
        elif (v, u) in Q:
            Q[(v, u)] += bias
        else:
            Q[(u, v)] = bias
    for (v, v), bias in iteritems(Q_min_color):
        if (v, v) in Q:
            Q[(v, v)] += bias
        else:
            Q[(v, v)] = bias

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # read off the coloring
    for v in x_vars:
        for c in x_vars[v]:
            if sample[x_vars[v][c]]:
                partial_coloring[v] = c

    # check that every node is colored
    for v in G:
        if v not in partial_coloring:
            # try to color it
            for c in possible_colors[v]:
                print v, c
            raise NotImplementedError

    return partial_coloring


def _minimum_coloring_qubo(x_vars, chi_lb, chi_ub, magnitude=1.):
    if chi_lb == chi_ub:
        # we know the chromatic number and there is no need for this
        return {}

    scaling = magnitude / (chi_ub - chi_lb)

    Q = {}
    for v in x_vars:
        for f, color in enumerate(range(chi_lb, chi_ub)):
            idx = x_vars[v][color]
            Q[(idx, idx)] = (f + 1) * scaling

    return Q


def _vertex_different_colors_qubo(G, x_vars):
    """For each vertex, it should not have the same color as any of its neighbors.

    Notes
    -----
        Does not enforce each node having a single color

        Ground energy is 0, infeasible gap is 1
    """
    Q = {}
    for u, v in G.edges_iter():
        if u not in x_vars or v not in x_vars:
            continue
        for color in x_vars[u]:
            if color in x_vars[v]:
                Q[(x_vars[u][color], x_vars[v][color])] = 1.
    return Q


def _vertex_one_color_qubo(x_vars):
    """For each vertex, it should have exactly one color

    Notes
    -----
        Does not enforce neighboring vertices having different colors.

        Ground energy is -1 * |G|, infeasible gap is 1
    """
    Q = {}
    for v in x_vars:
        for color in x_vars[v]:
            idx = x_vars[v][color]
            Q[(idx, idx)] = -1

        for color0, color1 in itertools.combinations(x_vars[v], 2):
            idx0 = x_vars[v][color0]
            idx1 = x_vars[v][color1]

            Q[(idx0, idx1)] = 2

    return Q


def _partial_precolor(G, chi_ub):
    """colors some of the nodes"""

    possible_colors = {v: set(range(chi_ub)) for v in G}

    # for now let's just pick an edge and color them
    u, v = next(iter(G.edges_iter()))
    partial_coloring = {u: 0, v: 1}
    chi_lb = 2  # lower bound for the chromatic number

    for w in G[u]:
        possible_colors[w].discard(0)
    for w in G[v]:
        possible_colors[w].discard(1)

    del possible_colors[u]
    del possible_colors[v]

    # TODO: I think if any variable has |colors| chi_ub - chi_lb we can wlog
    # assign it a color. Need to think more about this before implementing

    return partial_coloring, possible_colors, chi_lb


def _quadratic_chi_bound(n_edges):
    return int(math.ceil((1 + math.sqrt(1 + 8 * n_edges)) / 2))


def is_cycle(G):
    """Determines whether the given graph is a connected cycle graph.

    TODO

    https://en.wikipedia.org/wiki/Cycle_graph
    """
    trailing, leading = next(iter(G.edges()))
    start_node = trailing

    # travel around the graph, checking that each node has degree exactly two
    # also track how many nodes were visted
    n_visited = 1
    while leading != start_node:
        neighbors = G[leading]

        if len(neighbors) != 2:
            return False

        node1, node2 = neighbors

        if node1 == trailing:
            trailing, leading = leading, node2
        else:
            trailing, leading = leading, node1

        n_visited += 1

    # if we havent visited all of the nodes, then it is not a connected cycle
    return n_visited == len(G)


def is_vertex_coloring(G, coloring):
    """TODO"""
    return all(coloring[u] != coloring[v] for u, v in G.edges_iter())
