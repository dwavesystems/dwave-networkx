import itertools

from dwave_networkx.utils import binary_quadratic_model_sampler
from dwave_networkx import _PY2

__all__ = ['min_maximal_matching', 'is_matching', 'is_maximal_matching']

# compatibility for python 2/3
if _PY2:
    range = xrange
    iteritems = lambda d: d.iteritems()
    itervalues = lambda d: d.itervalues()
else:
    iteritems = lambda d: d.items()
    itervalues = lambda d: d.values()


@binary_quadratic_model_sampler(1)
def maximal_matching(G, sampler=None, **sampler_args):
    """Finds an approximate maximal matching.

    Defines a QUBO with ground states corresponding to a maximal
    matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule.

    Parameters
    ----------
    G : NetworkX graph

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
    matching : set
        A maximal matching of the graph.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    https://en.wikipedia.org/wiki/Matching_(graph_theory)

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    Based on the formulation presented in [AL]_

    """

    # the maximum degree
    delta = max(G.degree(node) for node in G)

    # use the maximum degree to determine the infeasible gaps
    A = 1.
    if delta == 2:
        B = .75
    else:
        B = .75 * A / (delta - 2.)  # we want A > (delta - 2) * B

    # each edge in G gets a variable, so let's create those
    edge_mapping = _edge_mapping(G)

    # build the QUBO
    Q = _maximal_matching_qubo(G, edge_mapping, magnitude=B)
    Qm = _matching_qubo(G, edge_mapping, magnitude=A)
    for edge, bias in Qm.items():
        if edge not in Q:
            Q[edge] = bias
        else:
            Q[edge] += bias

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # the matching are the edges that are 1 in the sample
    return set(edge for edge in G.edges if sample[edge_mapping[edge]] > 0)


@binary_quadratic_model_sampler(1)
def min_maximal_matching(G, sampler=None, **sampler_args):
    """Returns an approximate minimal maximal matching.

    Defines a QUBO with ground states corresponding to a minimal
    maximal matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule. A minimal maximal
    matching is the smallest maximal matching for G.

    Parameters
    ----------
    G : NetworkX graph

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
    matching : set
        A minimal maximal matching of the graph.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    https://en.wikipedia.org/wiki/Matching_(graph_theory)

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    .. [AL] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    # the maximum degree
    delta = max(G.degree(node) for node in G)

    # use the maximum degree to determine the infeasible gaps
    A = 1.
    if delta == 2:
        B = .75
    else:
        B = .75 * A / (delta - 2.)  # we want A > (delta - 2) * B
    C = .75 * B  # we want B > C

    # each edge in G gets a variable, so let's create those
    edge_mapping = _edge_mapping(G)

    # build the QUBO
    Q = _maximal_matching_qubo(G, edge_mapping, magnitude=B)
    Qm = _matching_qubo(G, edge_mapping, magnitude=A)
    for edge, bias in Qm.items():
        if edge not in Q:
            Q[edge] = bias
        else:
            Q[edge] += bias

    # to enforce the minimal constraint, we additionally add a small bias to
    # each variable
    for v in set(edge_mapping.values()):
        if (v, v) not in Q:
            Q[(v, v)] = C
        else:
            Q[(v, v)] += C

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # the matching are the edges that are 1 in the sample
    return set(edge for edge in G.edges if sample[edge_mapping[edge]] > 0)


def is_matching(edges):
    """Determines whether the given set of edges is a matching.

    A matching is a subset of edges in which no node occurs more than
    once.

    Parameters
    ----------
    edges : iterable
        A iterable of edges.

    Returns
    -------
    is_matching : bool
        True if the given edges are a matching.

    """
    return len(set().union(*edges)) == len(edges) * 2


def is_maximal_matching(G, matching):
    """Determines whether the given set of edges is a maximal matching.

    A matching is a subset of edges in which no node occurs more than
    once. The cardinality of a matching is the number of matched edges.
    A maximal matching is one where one cannot add any more edges
    without violating the matching rule.

    Parameters
    ----------
    G : NetworkX graph

    edges : iterable
        A iterable of edges.

    Returns
    -------
    is_matching : bool
        True if the given edges are a maximal matching.

    """
    touched_nodes = set().union(*matching)

    # first check if a matching
    if len(touched_nodes) != len(matching) * 2:
        return False

    # now for each edge, check that at least one of its variables is
    # already in the matching
    for (u, v) in G.edges:
        if u not in touched_nodes and v not in touched_nodes:
            return False

    return True


def _edge_mapping(G):
    """Assigns a variable for each edge in G.
    (u, v) and (v, u) map to the same variable.
    """
    edge_mapping = {edge: idx for idx, edge in enumerate(G.edges)}
    edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
    return edge_mapping


def _maximal_matching_qubo(G, edge_mapping, magnitude=1.):
    """Generates a QUBO that when combined with one as generated by _matching_qubo,
    induces a maximal matching on the given graph G.
    The variables in the QUBO are the edges, as given my edge_mapping.

    ground_energy = -1 * magnitude * |edges|
    infeasible_gap >= magnitude
    """
    Q = {}

    # for each node n in G, define a variable y_n to be 1 when n has a colored edge
    # and 0 otherwise.
    # for each edge (u, v) in the graph we want to enforce y_u OR y_v. This is because
    # if both y_u == 0 and y_v == 0, then we could add (u, v) to the matching.
    for (u, v) in G.edges:
        # 1 - y_v - y_u + y_v*y_u

        # for each edge connected to u
        for edge in G.edges(u):
            x = edge_mapping[edge]
            if (x, x) not in Q:
                Q[(x, x)] = -1 * magnitude
            else:
                Q[(x, x)] -= magnitude

        # for each edge connected to v
        for edge in G.edges(v):
            x = edge_mapping[edge]
            if (x, x) not in Q:
                Q[(x, x)] = -1 * magnitude
            else:
                Q[(x, x)] -= magnitude

        for e0 in G.edges(v):
            x0 = edge_mapping[e0]
            for e1 in G.edges(u):
                x1 = edge_mapping[e1]

                if x0 < x1:
                    if (x0, x1) not in Q:
                        Q[(x0, x1)] = magnitude
                    else:
                        Q[(x0, x1)] += magnitude
                else:
                    if (x1, x0) not in Q:
                        Q[(x1, x0)] = magnitude
                    else:
                        Q[(x1, x0)] += magnitude

    return Q


def _matching_qubo(G, edge_mapping, magnitude=1.):
    """Generates a QUBO that induces a matching on the given graph G.
    The variables in the QUBO are the edges, as given my edge_mapping.

    ground_energy = 0
    infeasible_gap = magnitude
    """
    Q = {}

    # We wish to enforce the behavior that no node has two colored edges
    for node in G:

        # for each pair of edges that contain node
        for edge0, edge1 in itertools.combinations(G.edges(node), 2):

            v0 = edge_mapping[edge0]
            v1 = edge_mapping[edge1]

            # penalize both being True
            Q[(v0, v1)] = magnitude

    return Q
