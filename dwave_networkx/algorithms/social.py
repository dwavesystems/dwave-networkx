from dwave_networkx.utils import binary_quadratic_model_sampler
from dwave_networkx import _PY2

__all__ = ["structural_imbalance"]

# compatibility for python 2/3
if _PY2:
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


@binary_quadratic_model_sampler(1)
def structural_imbalance(S, sampler=None, **sampler_args):
    """Returns an approximate set of frustrated edges and a bicoloring.

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
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrainted Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
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
    >>> S = nx.Graph()
    >>> S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
    >>> S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
    >>> S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
    >>> frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
    >>> print(frustrated_edges)
    {}
    >>> print(colors)  # doctest: +SKIP
    {'Alice': 0, 'Bob': 0, 'Eve': 1}
    >>> S.add_edge('Ted', 'Bob', sign=1)  # Ted is friendly with all
    >>> S.add_edge('Ted', 'Alice', sign=1)
    >>> S.add_edge('Ted', 'Eve', sign=1)
    >>> frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
    >>> print(frustrated_edges)
    {('Ted', 'Eve'): {'sign': 1}}
    >>> print(colors)  # doctest: +SKIP
    {'Bob': 1, 'Ted': 1, 'Alice': 1, 'Eve': 0}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    https://en.wikipedia.org/wiki/Ising_model

    References
    ----------
    .. [FIA] Facchetti, G., Iacono G., and Altafini C. (2011). Computing
       global structural balance in large-scale signed social networks.
       PNAS, 108, no. 52, 20953-20958

    """

    # format as an Ising problem
    h = {v: 0 for v in S}  # linear biases
    J = {}  # quadratic biases
    for u, v, data in S.edges(data=True):
        try:
            J[(u, v)] = -1. * data['sign']
        except KeyError:
            raise ValueError(("graph should be a signed social graph,"
                              "each edge should have a 'sign' attr"))

    # use the sampler to find low energy states
    response = sampler.sample_ising(h, J, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # spins determine the color
    colors = {v: (spin + 1) // 2 for v, spin in iteritems(sample)}

    # frustrated edges are the ones that are violated
    frustrated_edges = {}
    for u, v, data in S.edges(data=True):
        sign = data['sign']

        if sign > 0 and colors[u] != colors[v]:
            frustrated_edges[(u, v)] = data
        elif sign < 0 and colors[u] == colors[v]:
            frustrated_edges[(u, v)] = data
        # else: not frustrated or sign == 0, no relation to violate

    return frustrated_edges, colors
