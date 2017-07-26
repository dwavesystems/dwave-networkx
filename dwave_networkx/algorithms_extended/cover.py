from dwave_networkx.utils_dw import discrete_model_sampler

__all__ = ['min_vertex_cover_dm']


@discrete_model_sampler(1)
def min_vertex_cover_dm(G, sampler=None, **sampler_args):
    """Uses a discrete model sampler to determine a minimum vertex cover.

    A vertex cover is a set of verticies such that each edge of the graph
    is incident with at least one vertex in the set. A minimum vertex cover
    is the vertex cover of smallest size.

    Parameters
    ----------
    G : NetworkX graph

    sampler
        A discrete model sampler. A sampler is a process that samples
        from low energy states in models defined by an Ising equation
        or a Quadratic Unconstrainted Binary Optimization Problem
        (QUBO). A sampler is expected to have a 'sample_qubo' and
        'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    vertex_cover : list
       List of nodes that the form a the minimum vertex cover, as
       determined by the given sampler.

    Examples
    --------
    >>> G = dnx.chimera_graph(2, 2, 4)
    >>> dnx.min_vertex_cover_qa(G, Solver())
    [0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27]

    Notes
    -----
    Discrete model samplers by their nature may not return the lowest
    energy solution. This function does not attempt to confirm the
    quality of the returned sample.

    https://en.wikipedia.org/wiki/Vertex_cover

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    .. [1] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    # our weights for the two components. We need B < A
    A = 1  # term for ensuring that each edge has at least one node
    B = .5  # term for minimizing the number of nodes colored

    # ok, let's build the qubo. For each node n in the graph we assign a boolean variable
    # v_n where v_n = 1 when n is part of the vertex cover and v_n = 0 when n is not.

    # For each edge, we want at least one node to be colored. That is, for each edge
    # (n1, n2) we want a term in the hamiltonian A*(1 - v_n1)*(1 - v_n2). This can
    # be rewritten A*(1 - v_n1 - v_n2 + v_n1*v_n2). Since each edge contributes one
    # of these, our final hamiltonian is
    # H_a = A*(|E| + sum_(n) -(deg(n)*v_n) + sum_(n1,n2) v_n1*v_n2)
    # additionally, we want to have the minimum cover, so we add H_b = B*sum_(n) v_n
    Q = {(node, node): B - A * G.degree(node) for node in G}
    Q.update({edge: A for edge in G.edges_iter()})

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # nodes that are true are in the cover
    return [node for node in G if sample[node] > 0]
