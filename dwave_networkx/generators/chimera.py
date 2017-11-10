"""
Generators for some graphs derived from the D-Wave System.

"""
import networkx as nx
from networkx.algorithms.bipartite import color
from networkx import diameter

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException

__all__ = ['chimera_graph', 'find_chimera_indices']

# compatibility for python 2/3
if _PY2:
    range = xrange


def chimera_graph(m, n=None, t=None, create_using=None, node_list=None, edge_list=None, data=True):
    """Creates a Chimera lattice of size (m, n, t).

    A Chimera lattice is an m-by-n grid of Chimera tiles. Each Chimera
    tile is itself a bipartite graph with shores of size t. The
    connection in a Chimera lattice can be expressed using a node-indexing
    notation (i,j,u,k) for each node. (i,j) indexes the (row, column)
    of the Chimera tile. i must be between 0 and m-1, inclusive, and j
    must be between 0 and n-1, inclusive. u=0 indicates the left-hand
    nodes in the tile, and u=1 indicates the right-hand nodes.
    k=0,1,...,t-1 indexes nodes within either the left- or right-hand
    shores of a tile.

    In this notation, two nodes (i, j, u, k) and (i', j', u', k') are
    neighbors if and only if:

        (i = i' AND j = j' AND u != u') OR
        (i = i' +/- 1 AND j = j' AND u = 0 AND u' = 0 AND k = k') OR
        (i = i' AND j = j' +/- 1 AND u = 1 AND u' = 1 AND k = k')

    The first line gives the bipartite connections within the tile.
    The second and third give the vertical and horizontal connections
    between blocks respectively.

    Node (i, j, u, k) is labeled by:

        label = i * n * 2 * t + j * 2 * t + u * t + k

    Parameters
    ----------
    m : int
        The number of rows in the Chimera lattice.
    n : int, optional (default m)
        The number of columns in the Chimera lattice.
    t : int, optional (default 4)
        The size of the shore within each Chimera tile.
    create_using : Graph, optional (default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable, optional (default None)
        Iterable of nodes in the graph. If None, calculated from (m, n, t).
        Note that this list is used to remove nodes, so any nodes specified
        not in range(m * n * 2 * t) will not be added.
    edge_list : iterable, optional (default None)
        Iterable of edges in the graph. If None, edges are generated as
        described above. The nodes in each edge must be integer-labeled in
        range(m * n * t * 2).
    data : bool, optional (default True)
        If True, each node has a chimera_index attribute. The attribute
        is a 4-tuple Chimera index as defined above.

    Returns
    -------
    G : a NetworkX Graph
        An (m, n, t) Chimera lattice. Nodes are labeled by integers.

    Examples
    ========
    >>> G = dnx.chimera_graph(1, 1, 2)  # a single Chimera tile
    >>> len(G)
    4
    >>> list(G.nodes())
    [0, 1, 2, 3]
    >>> list(G.nodes(data=True))  # doctest: +SKIP
    [(0, {'chimera_index': (0, 0, 0, 0)}),
     (1, {'chimera_index': (0, 0, 0, 1)}),
     (2, {'chimera_index': (0, 0, 1, 0)}),
     (3, {'chimera_index': (0, 0, 1, 1)})]
    >>> list(G.edges())
    [(0, 2), (0, 3), (1, 2), (1, 3)]

    """

    if n is None:
        n = m
    if t is None:
        t = 4

    G = nx.empty_graph(0, create_using)

    G.name = "chimera_graph(%s, %s, %s)" % (m, n, t)

    max_size = m * n * 2 * t  # max number of nodes G can have

    if edge_list is None:
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
    else:
        G.add_edges_from(edge_list)

    if node_list is not None:
        nodes = set(node_list)
        G.remove_nodes_from(set(G) - nodes)
        G.add_nodes_from(nodes)  # for singleton nodes

    if data:
        v = 0
        for i in range(m):
            for j in range(n):
                for u in range(2):
                    for k in range(t):
                            if v in G:
                                G.node[v]['chimera_index'] = (i, j, u, k)
                            v += 1

    return G


def find_chimera_indices(G):
    """Tries to determine the Chimera indices of the nodes in G.

    See chimera_graph for a definition of a Chimera graph and Chimera
    indices.

    Only works for single-tile Chimera graphs.

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
    >>> chimera_indices = dnx.find_chimera_indices(G)

    >>> G = nx.Graph()
    >>> G.add_edges_from([(0, 2), (1, 2), (1, 3), (0, 3)])
    >>> chimera_indices = dnx.find_chimera_indices(G)
    >>> nx.set_node_attributes(G, 'chimera_index', chimera_indices)

    """

    # if the nodes are orderable, we want the lowest-order one.
    try:
        nlist = sorted(G.nodes)
    except TypeError:
        nlist = G.nodes()

    n_nodes = len(nlist)

    # create the object that will store the indices
    chimera_indices = {}

    # ok, let's first check for the simple cases
    if n_nodes == 0:
        return chimera_indices
    elif n_nodes == 1:
        raise DWaveNetworkXException('Singleton graphs are not Chimera-structured')
    elif n_nodes == 2:
        return {nlist[0]: (0, 0, 0, 0), nlist[1]: (0, 0, 1, 0)}

    # next, let's get the bicoloring of the graph; this raises an exception if the graph is
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

    raise Exception('not yet implemented for Chimera graphs with more than one tile')


def chimera_elimination_order(m, n=None, t=None):
    """Provides a variable elimination order for a Chimera graph.

    A graph defined by chimera_graph(m,n,t) has treewidth max(m,n)*t.
    This function outputs a variable elimination order inducing a tree
    decomposition of that width.

    Parameters
    ----------
    m : int
        The number of rows in the Chimera lattice.
    n : int, optional (default m)
        The number of columns in the Chimera lattice.
    t : int, optional (default 4)
        The size of the shore within each Chimera tile.


    Returns
    -------
    order : list
        An elimination order that induces the treewidth of chimera_graph(m,n,t).
    """
    if n is None:
        n = m

    if t is None:
        t = 4

    index_flip = m > n
    if index_flip:
        m,n = n,m

    def chimeraI(m0, n0, k0, l0):
        if index_flip:
            return m*2*t*n0 + 2*t*m0 + t*(1-k0) + l0
        else:
            return n*2*t*m0 + 2*t*n0 + t*k0 + l0

    order = []

    for n_i in range(n):
        for t_i in range(t):
            for m_i in range(m):
                order.append(chimeraI(m_i, n_i, 0, t_i))

    for n_i in range(n):
        for m_i in range(m):
            for t_i in range(t):
                order.append(chimeraI(m_i, n_i, 1, t_i))

    return order



