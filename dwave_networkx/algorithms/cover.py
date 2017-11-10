from dwave_networkx.algorithms.independent_set import maximum_independent_set
from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ['min_vertex_cover', 'is_vertex_cover']


@binary_quadratic_model_sampler(1)
def min_vertex_cover(G, sampler=None, **sampler_args):
    """Returns an approximate minimum vertex cover.

    Defines a QUBO with ground states corresponding to a minimum
    vertex cover and uses the sampler to sample from it.

    A vertex cover is a set of vertices such that each edge of the graph
    is incident with at least one vertex in the set. A minimum vertex cover
    is the vertex cover of smallest size.

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
    vertex_cover : list
       List of nodes that the form a the minimum vertex cover, as
       determined by the given sampler.

    Examples
    --------
    >>> G = dnx.chimera_graph(1, 2, 3)
    >>> dnx.min_vertex_cover(G, sampler)
    [0, 1, 2, 9, 10, 11]

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    https://en.wikipedia.org/wiki/Vertex_cover

    https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    References
    ----------
    .. [AL] Lucas, A. (2014). Ising formulations of many NP problems.
       Frontiers in Physics, Volume 2, Article 5.

    """
    indep_nodes = set(maximum_independent_set(G, sampler, **sampler_args))
    return [v for v in G if v not in indep_nodes]


def is_vertex_cover(G, vertex_cover):
    """Determines whether a given set of vertices is a cover.

    A vertex cover is a set of vertices such that each edge of the graph
    is incident with at least one vertex in the set.

    Parameters
    ----------
    G : NetworkX graph

    vertex_cover :
       Iterable of nodes that the form a the minimum vertex cover, as
       determined by the given sampler.

    Returns
    -------
    is_cover : bool
        True if the given iterable forms a vertex cover.

    """
    cover = set(vertex_cover)
    return all(u in cover or v in cover for u, v in G.edges)
