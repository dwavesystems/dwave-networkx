from dwave_networkx.exceptions import DWaveNetworkXException
from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["maximum_cut", "weighted_maximum_cut"]


@binary_quadratic_model_sampler(1)
def maximum_cut(G, sampler=None, **sampler_args):
    """Returns an approximate maximum cut.

    Defines an Ising problem with ground states corresponding to
    a maximum cut and uses the sampler to sample from it.

    A maximum cut is a subset S of the vertices of G such that
    the number of edges between S and the complementary subset
    is as large as possible.

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
    S : set
        A maximum cut of G.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # In order to form the Ising problem, we want to increase the
    # energy by 1 for each edge between two nodes of the same color.
    # The linear biases can all be 0.
    h = {v: 0. for v in G}
    J = {(u, v): 1 for u, v in G.edges}

    # draw the lowest energy sample from the sampler
    response = sampler.sample_ising(h, J, **sampler_args)
    sample = next(iter(response))

    return set(v for v in G if sample[v] >= 0)


def weighted_maximum_cut(G, sampler=None, **sampler_args):
    """Returns an approximate weighted maximum cut.

    Defines an Ising problem with ground states corresponding to
    a weighted maximum cut and uses the sampler to sample from it.

    A weighted maximum cut is a subset S of the vertices of G that
    maximizes the sum of the edge weights between S and its
    complementary subset.

    Parameters
    ----------
    G : NetworkX graph
        Each edge in G should have a numeric 'weight' attribute.

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
    S : set
        A maximum cut of G.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # In order to form the Ising problem, we want to increase the
    # energy by 1 for each edge between two nodes of the same color.
    # The linear biases can all be 0.
    h = {v: 0. for v in G}
    try:
        J = {(u, v): G[u][v]['weight'] for u, v in G.edges}
    except KeyError:
        raise DWaveNetworkXException("edges must have 'weight' attribute")

    # draw the lowest energy sample from the sampler
    response = sampler.sample_ising(h, J, **sampler_args)
    sample = next(iter(response))

    return set(v for v in G if sample[v] >= 0)
