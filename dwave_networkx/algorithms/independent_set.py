from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["maximum_independent_set", "is_independent_set"]


@binary_quadratic_model_sampler(1)
def maximum_independent_set(G, sampler=None, **sampler_args):
    """Returns an approximate maximum independent set.

    Defines a QUBO with ground states corresponding to a
    maximum independent set and uses the sampler to sample from
    it.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximum
    independent set is an independent set of largest possible size.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum cut independent set.

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
    indep_nodes : list
       List of nodes that form a maximum independent set, as
       determined by the given sampler.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a maximum
    independent set for a graph of a Chimera unit cell created using the
    `chimera_graph()` function.

    >>> import dimod
    >>> import dwave_networkx as dnx
    >>> samplerSA = dimod.SimulatedAnnealingSampler()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> dnx.maximum_independent_set(G, samplerSA)
    {4, 5, 6, 7}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Independent Set on Wikipedia <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_

    `QUBO on Wikipedia <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_

    .. [AL] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """

    # We assume that the sampler can handle an unstructured QUBO problem, so let's set one up.
    # Let us define the largest independent set to be S.
    # For each node n in the graph, we assign a boolean variable v_n, where v_n = 1 when n
    # is in S and v_n = 0 otherwise.
    # We call the matrix defining our QUBO problem Q.
    # On the diagnonal, we assign the linear bias for each node to be -1. This means that each
    # node is biased towards being in S
    # On the off diagnonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.
    Q = {(node, node): -1 for node in G}
    Q.update({edge: 2 for edge in G.edges})

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # nodes that are spin up or true are exactly the ones in S.
    return [node for node in sample if sample[node] > 0]


def is_independent_set(G, indep_nodes):
    """Determines whether the given nodes form an independent set.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges.

    Parameters
    ----------
    G : NetworkX graph
       The graph on which to check the independent set.

    indep_nodes : list
       List of nodes that form a maximum independent set, as
       determined by the given sampler.

    Returns
    -------
    is_independent : bool
        True if indep_nodes form an independent set.

    Example
    -------
    This example checks two sets of nodes, both derived from a
    single Chimera unit cell, for an independent set. The first set is
    the horizontal tile's nodes; the second has nodes from the horizontal and
    verical tiles.

    >>> import dwave_networkx as dnx
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> dnx.is_independent_set(G, [0, 1, 2, 3])
    True
    >>> dnx.is_independent_set(G, [0, 4])
    False

    """
    return not bool(G.subgraph(indep_nodes).edges)
