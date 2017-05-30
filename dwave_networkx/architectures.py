"""
Generators for some graphs derived from the D-Wave System.

"""
import sys

import dwave_networkx as dnx

__all__ = ['chimera_graph']

# compatibility for python 2/3
if sys.version_info[0] == 2:
    range = xrange


def chimera_graph(m, n=None, t=None, create_using=None):
    """Creates a Chimera lattice of size (m, n, t).

    A Chimera lattice is an m by n grid of Chimera Tiles. Each Chimera tile is itself a bipartite
    graph with shores of size t. The connection in a Chimera lattice can expressed using a node indexing
    notation (i,j,u,k) for each node. (i,j) indexes the (row, column) of the Chimera Tile. i must be
    between 0 and m-1 inclusive, and j must be between 0 and n-1 inclusive. u=0 indicates the left hand
    nodes in the tile, and u=1 indicates the right hand nodes. k=0,1,...,t-1 indexes nodes within either
    the left- or right-hand shores of a tile.
    In this notation, two nodes (i, j, u, k) and (i', j', u', k') are neighbors if and only if:
        (i = i' AND j = j' AND u != u') OR
        (i = i' +/- 1 AND j = j' AND u = 0 AND u' = 0 AND k = k') OR
        (i = i' AND j = j' +/- 1 AND u = 1 AND u' = 1 AND k = k')
    The first line gives the bipartite connections within the tile. The second and third give the
    vertical and horizontal connections between blocks respectively.

    Node (i, j, u, k) is labelled by:
        label = (i-1)*m + (j-1)*n + u*t + k

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

    Returns
    -------
    G : a networkx Graph
        An (m, n, t) Chimera lattice. Nodes are labelled by integers.

    Examples
    ========
    >>> G = dnx.chimera_graph(1, 1, 4)  # a single Chimera tile
    >>> len(G)
    8
    >>> list(G.nodes())
    [1, 2, 3, 4, 5, 6, 7]
    >>> list(G.edges())
    [(0, 4), (0, 5), (0, 6), (0, 7), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7),
    (3, 4), (3, 5), (3, 6), (3, 7)]

    """

    if n is None:
        n = m
    if t is None:
        t = 4

    G = dnx.empty_graph(m * n * t * 2, create_using)
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
