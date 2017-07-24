from __future__ import absolute_import

import sys

from dwave_networkx.utils_dw.decorators import discrete_model_sampler

__all__ = ["network_imbalance_dm"]

# compatibility for python 2/3
PY2 = sys.version_info[0] == 2
if PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


@discrete_model_sampler(1)
def network_imbalance_dm(S, sampler, **sampler_args):
    """Uses a discrete model sampler to determine the imbalance of
    the given social network.

    A signed social network graph is a graph whose signed edges
    represent friendly/hostile interactions between nodes. A
    signed social network is considered balanced if it can be cleanly
    divided into two factions, where all relations within a faction are
    friendly, and all relations between factions are hostile. The measure
    of imbalance or frustration is the minimum number of edges that
    violate this rule.

    Parameters
    ----------
    S : NetworkX graph
        Must be a social graph, that is each edge should have a 'sign'
        attribute with a numeric value.

    sampler
        A discrete model sampler. A sampler is a process that samples
        from low energy states in models defined by an Ising equation
        or a Quadratic Unconstrainted Binary Optimization Problem
        (QUBO). A sampler is expected to have a 'sample_qubo' and
        'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy.

    Additional keyword parameters are passed to the sampler.

    Returns
    -------
    frustrated_edges : dict
        A dictionary of the edges that violate the edge sign. The imbalance
        of the network is the length of frustrated_edges.

    colors: dict
        A bicoloring of the nodes into two factions.

    Raises
    ------
    ValueError
        If any edge does not have a 'sign' attribute.

    Examples
    --------
    >>> S = dnx.Graph()
    >>> S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
    >>> S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
    >>> S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
    >>> frustrated_edges, colors = dnx.network_imbalance_qubo(S, sampler)
    >>> print(frustrated_edges)
    {}
    >>> print(colors)
    {'Alice': 0, 'Bob': 0, 'Eve': 1}
    >>> S.add_edge('Ted', 'Bob', sign=1)  # Ted is friendly with all
    >>> S.add_edge('Ted', 'Alice', sign=1)
    >>> S.add_edge('Ted', 'Eve', sign=1)
    >>> frustrated_edges, colors = dnx.network_imbalance_qubo(S, sampler)
    >>> print(frustrated_edges)
    {('Ted', 'Eve'): {'sign': 1}}
    >>> print(colors)
    {'Bob': 1, 'Ted': 1, 'Alice': 1, 'Eve': 0}

    Notes
    -----
    Discrete model samplers by their nature may not return the lowest
    energy solution. This function does not attempt to confirm the
    quality of the returned sample.

    https://en.wikipedia.org/wiki/Ising_model

    References
    ----------
    .. [1] Facchetti, G., Iacono G., and Altafini C. (2011). Computing
       global structural balance in large-scale signed social networks.
       PNAS, 108, no. 52, 20953-20958

    """

    # format as an Ising problem
    h = {v: 0 for v in S}  # linear biases
    J = {}  # quadratic biases
    for u, v, data in S.edges_iter(data=True):
        try:
            J[(u, v)] = -1. * data['sign']
        except KeyError:
            raise ValueError(("graph should be a signed social graph,"
                              "each edge should have a 'sign' attr"))

    # put the problem on the sampler
    result = sampler.sample_ising(h, J, **sampler_args)

    # get the lowest energy sample
    sample = next(iter(result))

    # spins determine the color
    colors = {v: (spin + 1) // 2 for v, spin in iteritems(sample)}

    # frustrated edges are the ones that are violated
    frustrated_edges = {}
    for u, v, data in S.edges_iter(data=True):
        sign = data['sign']

        if sign > 0 and colors[u] != colors[v]:
            frustrated_edges[(u, v)] = data
        elif sign < 0 and colors[u] == colors[v]:
            frustrated_edges[(u, v)] = data
        else:
            # sign == 0, no relation to violate
            pass

    return frustrated_edges, colors
