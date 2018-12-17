from __future__ import division
from dwave_networkx.utils import binary_quadratic_model_sampler
import networkx as nx

__all__ = ["traveling_salesman", "traveling_salesman_qubo", "is_hamiltonian_path"]

@binary_quadratic_model_sampler(1)
def traveling_salesman(G, sampler=None, lagrange=2.0, **sampler_args):
    """Returns an approximate minimum traveling salesperson route.
    Defines a QUBO with ground states corresponding to a
    minimum route and uses the sampler to sample
    from it.

    A route is a cycle in the graph that reaches each node exactly once.
    A minimum route is a route with the smallest total edge weight.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum traveling salesperson route.
        This should be a complete graph with non-zero weights on every edge.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    lagrange : optional (default 2)
        Lagrange parameter to weight constraints (visit every city once)
        versus objective (shortest distance route).

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    route : list
       List of nodes in order to be visited on a route

    Examples
    --------
    This example uses a `dimod <https://github.com/dwavesystems/dimod>`_ sampler
    to find a minimum route in a five-cities problem.

    >>> import dwave_networkx as dnx
    >>> import networkx as nx
    >>> import dimod
    ...
    >>> G = nx.complete_graph(4)
    >>> G.add_weighted_edges_from({(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 3), (1, 3, 4), (2, 3, 5)})
    >>> dnx.traveling_salesman(G, dimod.ExactSolver())
    [2, 1, 0, 3]

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # Get a QUBO representation of the problem
    Q = traveling_salesman_qubo(G, lagrange)

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample, in order by stop number
    sample = next(iter(response))
    route = []
    for entry in sample:
        if sample[entry] > 0:
            route.append(entry)
    route.sort(key=lambda x: x[1])
    route = ( x[0] for x in route)
    return list(route)

def traveling_salesman_qubo(G, lagrange=2.0):
    """Return the QUBO with ground states corresponding to a minimum TSP route.

    Parameters
    ----------
    G : NetworkX graph
        Nodes in graph must be labeled 0...N-1

    lagrange : optional (default 2)
        Lagrange parameter to weight constraints (no edges within set)
        versus objective (largest set possible).

    Returns
    -------
    QUBO : dict
       The QUBO with ground states corresponding to a maximum weighted independent set.
    """

    # empty QUBO for an empty graph
    if not G:
        return {}

    N = G.number_of_nodes()

    ## Creating the QUBO
    # Start with an empty QUBO
    Q = {}
    Q = {((node_1,pos_1),(node_2,pos_2)): 0.0 for node_1 in G for node_2 in G for pos_1 in range(N) for pos_2 in range(N)}

    # Constraint that each row has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += 2.0*lagrange

    # Constraint that each col has exactly one 1
    for pos in range(N):
        for node_1 in G:
            Q[((node_1, pos), (node_1,pos))] -= lagrange
            for node_2 in set(G)-{node_1}:
                Q[((node_1, pos), (node_2,pos))] += 2.0*lagrange

    # Objective that minimizes distance
    for node_1 in G:
        for node_2 in G:
            if node_1<node_2:
                for pos in range(N):
                    Q[((node_1,pos), (node_2,(pos+1)%N))] += G[node_1][node_2]['weight']

    return Q

def is_hamiltonian_path(G, route):
    """Determines whether the given list forms a valid TSP route.

    A TSP route must visit each city exactly once.

    Parameters
    ----------
    G : NetworkX graph

        The graph on which to check the route.

    route : list

        List of nodes in the order that they are visited.

    Returns
    -------
    is_valid : bool

    True if route forms a valid TSP route.
    """

    return (len(route) == len(set(G)))
