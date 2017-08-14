"""
Generators for some graphs derived from the D-Wave System.

"""
import itertools

import networkx as nx
from networkx.algorithms.bipartite import color
from networkx import diameter

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException

__all__ = ['chimera_graph', 'find_chimera_indices']

# compatibility for python 2/3
if _PY2:
    range = xrange


def chimera_graph(m, n=None, t=None, create_using=None, data=True):
    """Creates a Chimera lattice of size (m, n, t).

    A Chimera lattice is an m by n grid of Chimera Tiles. Each Chimera
    tile is itself a bipartite graph with shores of size t. The
    connection in a Chimera lattice can expressed using a node indexing
    notation (i,j,u,k) for each node. (i,j) indexes the (row, column)
    of the Chimera Tile. i must be between 0 and m-1 inclusive, and j
    must be between 0 and n-1 inclusive. u=0 indicates the left hand
    nodes in the tile, and u=1 indicates the right hand nodes.
    k=0,1,...,t-1 indexes nodes within either the left- or right-hand
    shores of a tile.

    In this notation, two nodes (i, j, u, k) and (i', j', u', k') are
    neighbors if and only if::

        (i = i' AND j = j' AND u != u') OR
        (i = i' +/- 1 AND j = j' AND u = 0 AND u' = 0 AND k = k') OR
        (i = i' AND j = j' +/- 1 AND u = 1 AND u' = 1 AND k = k')

    The first line gives the bipartite connections within the tile.
    The second and third give the vertical and horizontal connections
    between blocks respectively.

    Node (i, j, u, k) is labelled by:

        label = i * m + j * n + u * t + k

    Parameters
    ----------
    m : int
        The number of rows in the Chimera lattice.
    n : int, optional (default m)
        The number of columns in the Chimera lattice.
    t : int, optional (default 4)
        The size of the shore within each Chimera tile.
    create_using : Graph, optional (default None)
        If provided this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    data : bool, optional (default True)
        If True, each node has a chimera_index attribute. The attribute's
        value is 4-tuple Chimera index as defined above.


    Returns
    -------
    G : a NetworkX Graph
        An (m, n, t) Chimera lattice. Nodes are labelled by integers.

    Examples
    ========
    >>> G = dnx.chimera_graph(1, 1, 4)  # a single Chimera tile
    >>> len(G)
    8
    >>> list(G.nodes())
    [1, 2, 3, 4, 5, 6, 7]
    >>> list(G.nodes(data=True))
    [(0, {'chimera_index': (0, 0, 0, 0)}),
     (1, {'chimera_index': (0, 0, 0, 1)}),
     (2, {'chimera_index': (0, 0, 0, 2)}),
     (3, {'chimera_index': (0, 0, 0, 3)}),
     (4, {'chimera_index': (0, 0, 1, 0)}),
     (5, {'chimera_index': (0, 0, 1, 1)}),
     (6, {'chimera_index': (0, 0, 1, 2)}),
     (7, {'chimera_index': (0, 0, 1, 3)})]
    >>> list(G.edges())
    [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 5), (3, 6), (3, 7)]

    """

    if n is None:
        n = m
    if t is None:
        t = 4

    if data:
        G = nx.empty_graph(0, create_using)
        label = 0
        for i in range(m):
            for j in range(n):
                for u in range(2):
                    for k in range(t):
                        G.add_node(label, chimera_index=(i, j, u, k))
                        label += 1
    else:
        G = nx.empty_graph(m * n * 2 * t, create_using)

    G.name = "chimera_graph(%s, %s, %s)" % (m, n, t)

    hoff = 2 * t
    voff = n * hoff
    mi = m * voff
    ni = n * hoff

    # tile edges
    G.add_edges_from((k0, k1)
                     for i in range(0, ni, hoff)
                     for j in range(i, mi, voff)
                     for k0 in range(j, j + t)
                     for k1 in range(j + t, j + 2 * t))
    # horizontal edges
    G.add_edges_from((k, k + hoff)
                     for i in range(t, 2 * t)
                     for j in range(i, ni - hoff, hoff)
                     for k in range(j, mi, voff))
    # vertical edges
    G.add_edges_from((k, k + voff)
                     for i in range(t)
                     for j in range(i, ni, hoff)
                     for k in range(j, mi - voff, voff))

    return G


def find_chimera_indices(G):
    """Tries to determine the Chimera indices of the nodes in G.

    See chimera_graph for a definition of a Chimera graph and Chimera
    indices.

    Currently only works for single tile Chimera graphs.

    Parameters
    ----------
    G : a NetworkX graph.

    Returns
    -------
    chimera_indices : dict
        A dict of the form {node: (i, j, u, k), ...} where (i, j, u, k)
        is a 4-tuple of integer Chimera indices.

    Examples
    --------
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> chimera_indices = find_chimera_indices(G)

    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 2), (1, 2), (1, 3), (0, 3)])
    >>> chimera_indices = dnx.find_chimera_indices(G)
    >>> nx.set_node_attributes(G, 'chimera_index', chimera_indices)

    """

    # if the nodes are orderable, we want the lowest order one.
    try:
        nlist = sorted(G.nodes_iter())
    except TypeError:
        nlist = G.nodes()

    n_nodes = len(nlist)

    # create the object that will store the indices
    chimera_indices = {}

    # ok, let's first check for the simple cases
    if n_nodes == 0:
        return chimera_indices
    elif n_nodes == 1:
        raise DWaveNetworkXException('Singleton graphs are not Chimera-stuructured')
    elif n_nodes == 2:
        return {nlist[0]: (0, 0, 0, 0), nlist[1]: (0, 0, 1, 0)}

    # next, let's get the bicoloring of the graph, this raises an exception of the graph is
    # not bipartite
    coloring = color(G)

    # we want the color of the node to be the u term in the Chimera-index, so we want the
    # first node in nlist to be color 0
    if coloring[nlist[0]] == 1:
        coloring = {v: 1 - coloring[v] for v in coloring}

    # we also want the diameter of the graph
    # claim: diameter(G) == m + n for |G| > 2
    dia = diameter(G)

    # we have already handled the |G| <= 2 case, so we know, for diameter == 2, that the Chimera
    # graph is a single tile
    if dia == 2:
        shore_indices = [0, 0]

        for v in nlist:
            u = coloring[v]
            chimera_indices[v] = (0, 0, u, shore_indices[u])
            shore_indices[u] += 1

        return chimera_indices

    # NB: max degree == shore size <==> one tile

    raise NotImplementedError('not yet implemented for Chimera graphs with more than one tile')
