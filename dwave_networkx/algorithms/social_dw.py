"""
TODO
"""
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
def network_imbalance_dm(S, sampler, **solver_args):
    """Determines the imbalance of the given social network.

    Parameters
    ----------
    S : NetworkX graph
        Must be a social graph, that is each edge should have a 'sign'
        attribute.

    sampler
        TODO

    Additional keyword parameters are passed to the given solver.

    Returns
    -------
    frustrated_edges : dict
        A dictionary of the edges that violate the edge sign. The imbalance
        of the network is the length of frustrated_edges.

    colors: dict
        A bicoloring of the nodes into two teams.

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
    result = sampler.sample_ising(h, J)

    # get the lowest energy sample
    sample, energy = next(result.items())

    # spins determine the color
    colors = {v: (spin + 1) / 2 for v, spin in iteritems(sample)}

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
