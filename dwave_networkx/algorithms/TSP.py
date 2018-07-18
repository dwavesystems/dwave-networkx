from __future__ import division

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["traveling_salesman", "traveling_salesman_qubo", "is_valid_route"]

##@binary_quadratic_model_sampler(2)
def traveling_salesman(G, sampler=None, lagrange=2.0, **sampler_args):
    """Returns an approximate minimum traveling salesman route.
    Defines a QUBO with ground states corresponding to a
    minimum route and uses the sampler to sample
    from it.

    A route is a cycle in the graph that reaches each node exactly once. 
    A minimum route is a route with the smallest total edge weight.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum cut weighted independent set.

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
        Lagrange parameter to weight constraints (no edges within set) 
        versus objective (largest set possible).

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    route : list
       List of nodes in order to be visited on a route

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    """
    # Get a QUBO representation of the problem
    Q = traveling_salesman_qubo(G, lagrange)

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # Define route for solution found
    N = len(G)
    route = [-1]*N
    route[0]=0
    for node in sample:
        if sample[node]>0:
            j = node%N
            v = (node-j)/N
            route[j] = int(v)

    # nodes that are spin up or true are exactly the ones in S.
    return route

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

    N = len(G)

    def index(a, b):
        return (a)*N+(b)

    ## Creating the QUBO
    # Start with an empty QUBO
    Q = {}
    for i in range(N*N):
        for j in range(N*N):
            Q.update({(i,j): 0})

    # Constraint that each row has exactly one 1
    for v in range(N):
        for j in range(N):
            Q[(index(v,j), index(v,j))] += -1*lagrange
            for k in range(j+1, N):
                Q[(index(v,j), index(v,k))] += 2*lagrange
                Q[(index(v,k), index(v,j))] += 2*lagrange

    # Constraint that each col has exactly one 1
    for j in range(N):
        for v in range(N):
            Q[(index(v,j), index(v,j))] += -1*lagrange
            for w in range(v+1,N):
                Q[(index(v,j), index(w,j))] += 2*lagrange
                Q[(index(w,j), index(v,j))] += 2*lagrange

    # Objective that minimizes distance
    for u in range(N):
        for v in range(N):
            if u!=v:
                for j in range(N):
                    Q[(index(u,j), index(v,(j+1)%N))] += G[u][v]['weight']

    return Q

def is_valid_route(G, route):
    """Determines whether the given list forms a valid TSP route.

    An TSP route must visit each city exactly once.

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

    return (-1 not in route) and (sum(route) == sum(range(len(G))))