from __future__ import division

import math
import itertools

import networkx as nx
from dwave_networkx import _PY2
from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["min_vertex_coloring", "is_vertex_coloring", "is_cycle"]

# compatibility for python 2/3
if _PY2:
    range = xrange

    def iteritems(d): return d.iteritems()

    def ceil(n): return int(math.ceil(n))
else:
    def iteritems(d): return d.items()
    ceil = math.ceil

try:
    import numpy
    eigenvalues = numpy.linalg.eigvals
except ImportError:
    eigenvalues = False


@binary_quadratic_model_sampler(1)
def min_vertex_coloring(G, sampler=None, **sampler_args):
    """Returns an approximate minimum vertex coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color. A minimum vertex coloring is the problem of solving
    the vertex coloring problem using the smallest number of colors.

    Since neighboring vertices must satisfy a constraint of having
    different colors, the problem can be posed as a binary constraint
    satisfaction problem.

    Defines a QUBO with ground states corresponding to minimum
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    coloring : dict
        A coloring for each vertex in G such that no adjacent nodes
        share the same color. A dict of the form {node: color, ...}

    Example
    -------
    This example colors a single Chimera unit cell. It colors the four
    horizontal qubits one color (0) and the four vertical qubits another (1).

    >>> # Set up a sampler; this example uses a sampler from dimod https://github.com/dwavesystems/dimod
    >>> import dimod
    >>> import dwave_networkx as dnx
    >>> samplerSA = dimod.SimulatedAnnealingSampler()
    >>> # Create a graph and color it
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> colors = dnx.min_vertex_coloring(G, sampler=samplerSA)
    >>> colors
    {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

    References
    ----------
    .. [DWMP] Dahl, E., "Programming the D-Wave: Map Coloring Problem",
       https://www.dwavesys.com/sites/default/files/Map%20Coloring%20WP2.pdf

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    # if the given graph is not connected, apply the function to each connected component
    # seperately.
    if not nx.is_connected(G):
        coloring = {}
        for subG in nx.connected_component_subgraphs(G):
            sub_coloring = min_vertex_coloring(subG, sampler, **sampler_args)
            coloring.update(sub_coloring)
        return coloring

    n_nodes = len(G)  # number of nodes
    n_edges = len(G.edges)  # number of edges

    # ok, first up, we can eliminate a few graph types trivially

    # Graphs with no edges, have chromatic number 1
    if not n_edges:
        return {node: 0 for node in G}

    # Complete graphs have chromatic number N
    if n_edges == n_nodes * (n_nodes - 1) // 2:
        return {node: color for color, node in enumerate(G)}

    # The number of variables in the QUBO is approximately the number of nodes in the graph
    # times the number of potential colors, so we want as tight an upper bound on the
    # chromatic number (chi) as possible
    chi_ub = _chromatic_number_upper_bound(G, n_nodes, n_edges)

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

    # combine all three constraints
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

    return partial_coloring


def _chromatic_number_upper_bound(G, n_nodes, n_edges):
    # tries to determine an upper bound on the chromatic number of G
    # Assumes G is not complete

    # chi * (chi - 1) <= 2 * |E|
    quad_bound = ceil((1 + math.sqrt(1 + 8 * n_edges)) / 2)

    if n_nodes % 2 == 1 and is_cycle(G):
        # odd cycle graphs need three colors
        bound = 3
    else:
        if not eigenvalues:
            # chi <= max degree, unless it is complete or a cycle graph of odd length,
            # in which case chi <= max degree + 1 (Brook's Theorem)
            bound = max(G.degree(node) for node in G)
        else:
            # Let A be the adj matrix of G (symmetric, 0 on diag). Let theta_1
            # be the largest eigenvalue of A. Then chi <= theta_1 + 1 with
            # equality iff G is complete or an odd cycle.
            # this is strictly better than brooks theorem
            bound = ceil(max(eigenvalues(nx.to_numpy_matrix(G))))

    return min(quad_bound, bound)


def _minimum_coloring_qubo(x_vars, chi_lb, chi_ub, magnitude=1.):
    """We want to disincentivize unneeded colors. Generates the QUBO
    that does that.
    """
    # if we already know the chromatic number, then we don't need to
    # disincentivize any colors.
    if chi_lb == chi_ub:
        return {}

    # we might need to use some of the colors, so we want to disincentivize
    # them in increasing amounts, linearly.
    scaling = magnitude / (chi_ub - chi_lb)

    # build the QUBO
    Q = {}
    for v in x_vars:
        for f, color in enumerate(range(chi_lb, chi_ub)):
            idx = x_vars[v][color]
            Q[(idx, idx)] = (f + 1) * scaling

    return Q


def _vertex_different_colors_qubo(G, x_vars):
    """For each vertex, it should not have the same color as any of its
    neighbors. Generates the QUBO to enforce this constraint.

    Notes
    -----
    Does not enforce each node having a single color.

    Ground energy is 0, infeasible gap is 1.
    """
    Q = {}
    for u, v in G.edges:
        if u not in x_vars or v not in x_vars:
            continue
        for color in x_vars[u]:
            if color in x_vars[v]:
                Q[(x_vars[u][color], x_vars[v][color])] = 1.
    return Q


def _vertex_one_color_qubo(x_vars):
    """For each vertex, it should have exactly one color. Generates
    the QUBO to enforce this constraint.

    Notes
    -----
    Does not enforce neighboring vertices having different colors.

    Ground energy is -1 * |G|, infeasible gap is 1.
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
    """In order to reduce the number of variables in the QUBO, we want to
    color as many nodes as possible without affecting the min vertex
    coloring. Without loss of generality, we can choose a single maximal
    clique and color each node in it uniquely.

    Returns
    -------
        partial_coloring : dict
        A dict describing a partial coloring of the nodes of G. Of the form
        {node: color, ...}.

        possible_colors : dict
        A dict giving the possible colors for each node in G not already
        colored. Of the form {node: set([color, ...]), ...}.

        chi_lb : int
        A lower bound on the chromatic number chi.

    Notes
    -----
        partial_coloring.keys() and possible_colors.keys() should be
        disjoint.

    """

    # find a random maximal clique and give each node in it a unique color
    v = next(iter(G))
    clique = [v]
    for u in G[v]:
        if all(w in G[u] for w in clique):
            clique.append(u)

    partial_coloring = {v: c for c, v in enumerate(clique)}
    chi_lb = len(partial_coloring)  # lower bound for the chromatic number

    # now for each uncolored node determine the possible colors
    possible_colors = {v: set(range(chi_ub)) for v in G if v not in partial_coloring}

    for v, color in iteritems(partial_coloring):
        for u in G[v]:
            if u in possible_colors:
                possible_colors[u].discard(color)

    # TODO: there is more here that can be done. For instance some nodes now
    # might only have one possible color. Or there might only be one node
    # remaining to color

    return partial_coloring, possible_colors, chi_lb


def is_cycle(G):
    """Determines whether the given graph is a cycle or circle graph.

    A cycle graph or circular graph is a graph that consists of a single cycle.

    https://en.wikipedia.org/wiki/Cycle_graph

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    is_cycle : bool
        True if the graph consists of a single cycle.

    """
    trailing, leading = next(iter(G.edges))
    start_node = trailing

    # travel around the graph, checking that each node has degree exactly two
    # also track how many nodes were visited
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

    # if we haven't visited all of the nodes, then it is not a connected cycle
    return n_visited == len(G)


def is_vertex_coloring(G, coloring):
    """Determines whether the given coloring is a vertex coloring of graph G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which the vertex coloring is applied.

    coloring : dict
        A coloring of the nodes of G. Should be a dict of the form
        {node: color, ...}.

    Returns
    -------
    is_vertex_coloring : bool
        True if the given coloring defines a vertex coloring; that is, no
        two adjacent vertices share a color.

    Example
    -------
    This example colors checks two colorings for a graph, G, of a single Chimera
    unit cell. The first uses one color (0) for the four horizontal qubits
    and another (1) for the four vertical qubits, in which case there are
    no adjacencies; the second coloring swaps the color of one node.

    >>> G = dnx.chimera_graph(1,1,4)
    >>> colors = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    >>> dnx.is_vertex_coloring(G, colors)
    True
    >>> colors[4]=0
    >>> dnx.is_vertex_coloring(G, colors)
    False

   """
    return all(coloring[u] != coloring[v] for u, v in G.edges)
