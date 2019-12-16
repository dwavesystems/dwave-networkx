# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""
Generators for graphs derived from the D-Wave System.

"""
import warnings

import networkx as nx
from networkx.algorithms.bipartite import color
from networkx import diameter

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException

__all__ = ['chimera_graph',
           'chimera_coordinates',
           'find_chimera_indices',
           'chimera_to_linear',
           'linear_to_chimera']

# compatibility for python 2/3
if _PY2:
    range = xrange


def chimera_graph(m, n=None, t=None, create_using=None, node_list=None, edge_list=None, data=True, coordinates=False):
    """Creates a Chimera lattice of size (m, n, t).

    A Chimera lattice is an m-by-n grid of Chimera tiles. Each Chimera
    tile is itself a bipartite graph with shores of size t. The
    connection in a Chimera lattice can be expressed using a node-indexing
    notation (i,j,u,k) for each node.

    * (i,j) indexes the (row, column) of the Chimera tile. i must be
      between 0 and m-1, inclusive, and j must be between 0 and
      n-1, inclusive.
    * u=0 indicates the left-hand nodes in the tile, and u=1 indicates
      the right-hand nodes.
    * k=0,1,...,t-1 indexes nodes within either the left- or
      right-hand shores of a tile.

    In this notation, two nodes (i, j, u, k) and (i', j', u', k') are
    neighbors if and only if:

        (i = i' AND j = j' AND u != u') OR
        (i = i' +/- 1 AND j = j' AND u = 0 AND u' = 0 AND k = k') OR
        (i = i' AND j = j' +/- 1 AND u = 1 AND u' = 1 AND k = k')

    The first of the three terms of the disjunction gives the
    bipartite connections  within the tile. The second and third terms
    give the vertical and horizontal connections between blocks
    respectively.

    Node (i, j, u, k) is labeled by:

        label = i * n * 2 * t + j * 2 * t + u * t + k

    Parameters
    ----------
    m : int
        Number of rows in the Chimera lattice.
    n : int (optional, default m)
        Number of columns in the Chimera lattice.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.
    create_using : Graph (optional, default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph. If None, calculated
        from (m, n, t). Note that this list is used to remove nodes,
        so any nodes specified not in `range(m * n * 2 * t)` are not added.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. If None, edges are
        generated as described above. The nodes in each edge must be
        integer-labeled in `range(m * n * t * 2)`.
    data : bool (optional, default True)
        If True, each node has a
        `chimera_index attribute`. The attribute is a 4-tuple Chimera index
        as defined above.
    coordinates : bool (optional, default False)
        If True, node labels are 4-tuples, equivalent to the chimera_index
        attribute as above.  In this case, the `data` parameter controls the
        existence of a `linear_index attribute`, which is an int

    Returns
    -------
    G : NetworkX Graph
        An (m, n, t) Chimera lattice. Nodes are labeled by integers.

    Examples
    ========
    >>> G = dnx.chimera_graph(1, 1, 2)  # a single Chimera tile
    >>> len(G)
    4
    >>> list(G.nodes())  # doctest: +SKIP
    [0, 1, 2, 3]
    >>> list(G.nodes(data=True))  # doctest: +SKIP
    [(0, {'chimera_index': (0, 0, 0, 0)}),
     (1, {'chimera_index': (0, 0, 0, 1)}),
     (2, {'chimera_index': (0, 0, 1, 0)}),
     (3, {'chimera_index': (0, 0, 1, 1)})]
    >>> list(G.edges())  # doctest: +SKIP
    [(0, 2), (0, 3), (1, 2), (1, 3)]

    """
    m = int(m)
    if n is None:
        n = m
    else:
        n = int(n)
    if t is None:
        t = 4
    else:
        t = int(t)

    G = nx.empty_graph(0, create_using)

    G.name = "chimera_graph(%s, %s, %s)" % (m, n, t)

    construction = (("family", "chimera"), ("rows", m), ("columns", n),
                    ("tile", t), ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    G.graph.update(construction)

    max_size = m * n * 2 * t  # max number of nodes G can have

    if edge_list is None:
        if coordinates:
            # tile edges
            G.add_edges_from(((i, j, 0, k0), (i, j, 1, k1))
                             for i in range(m)
                             for j in range(n)
                             for k0 in range(t)
                             for k1 in range(t))

            # horizontal edges
            G.add_edges_from(((i, j, 1, k), (i, j+1, 1, k))
                             for i in range(m)
                             for j in range(n-1)
                             for k in range(t))
            # vertical edges
            G.add_edges_from(((i, j, 0, k), (i+1, j, 0, k))
                             for i in range(m-1)
                             for j in range(n)
                             for k in range(t))
        else:
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
        if coordinates:
            def checkadd(v, q):
                if q in G:
                    G.nodes[q]['linear_index'] = v
        else:
            def checkadd(v, q):
                if v in G:
                    G.nodes[v]['chimera_index'] = q

        v = 0
        for i in range(m):
            for j in range(n):
                for u in range(2):
                    for k in range(t):
                        checkadd(v, (i, j, u, k))
                        v += 1

    return G


def find_chimera_indices(G):
    """Attempts to determine the Chimera indices of the nodes in graph G.

    See the `chimera_graph()` function for a definition of a Chimera graph and Chimera
    indices.

    Parameters
    ----------
    G : NetworkX graph
        Should be a single-tile Chimera graph.

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
    >>> nx.set_node_attributes(G, chimera_indices, 'chimera_index')

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
        raise DWaveNetworkXException(
            'Singleton graphs are not Chimera-structured')
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


class chimera_coordinates(object):
    def __init__(self, m, n=None, t=None):
        """Provides coordinate converters for the chimera indexing scheme.

        Parameters
        ----------
        m : int
            The number of rows in the Chimera lattice.
        n : int, optional (default m)
            The number of columns in the Chimera lattice.
        t : int, optional (default 4)
            The size of the shore within each Chimera tile.

        Examples
        --------

        Convert between Chimera coordinates and linear indices directly

        >>> coords = dnx.chimera_coordinates(16, 16, 4)
        >>> coords.chimera_to_linear((0, 2, 0, 1))
        17
        >>> coords.linear_to_chimera(17)
        (0, 2, 0, 1)

        Construct a new graph with the coordinate labels

        >>> C16 = dnx.chimera_graph(16)
        >>> coords = dnx.chimera_coordinates(16)
        >>> G = nx.Graph()
        >>> G.add_nodes_from(coords.iter_linear_to_chimera(C16.nodes))
        >>> G.add_edges_from(coords.iter_linear_to_chimera_pairs(C16.edges))

        See also
        --------
        :func:`.chimera_graph` for a description of the various conventions.

        """
        self.args = m, m if n is None else n, 4 if t is None else t

    def int(self, q):
        """Deprecated alias for `chimera_to_linear`."""
        msg = ('chimera_coordinates.int is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.chimera_to_linear instead')
        warnings.warn(msg, DeprecationWarning)
        return self.chimera_to_linear(q)

    def chimera_to_linear(self, q):
        """Convert a 4-term Chimera coordinate to a linear index."""
        i, j, u, k = q
        m, n, t = self.args
        return ((n*i + j)*2 + u)*t + k

    def tuple(self, r):
        """Deprecated alias for `linear_to_chimera`."""
        msg = ('chimera_coordinates.tuple is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.linear_to_chimera instead')
        warnings.warn(msg, DeprecationWarning)
        return self.linear_to_chimera(r)

    def linear_to_chimera(self, r):
        """Convert a linear index to a 4-term Chimera coordinate."""
        m, n, t = self.args
        r, k = divmod(r, t)
        r, u = divmod(r, 2)
        i, j = divmod(r, n)
        return i, j, u, k

    def ints(self, qlist):
        """Deprecated alias for `iter_chimera_to_linear`."""
        msg = ('chimera_coordinates.ints is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.iter_chimera_to_linear instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_chimera_to_linear(qlist)

    def iter_chimera_to_linear(self, qlist):
        """Return an iterator converting a sequence of 4-term Chimera
        coordinates to linear indices.
        """
        m, n, t = self.args
        for (i, j, u, k) in qlist:
            yield ((n*i + j)*2 + u)*t + k

    def tuples(self, rlist):
        """Deprecated alias for `iter_linear_to_chimera`."""
        msg = ('chimera_coordinates.tuples is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.iter_linear_to_chimera instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_linear_to_chimera(rlist)

    def iter_linear_to_chimera(self, rlist):
        """Return an iterator converting a sequence of linear indices to 4-term
        Chimera coordinates.
        """
        m, n, t = self.args
        for r in rlist:
            r, k = divmod(r, t)
            r, u = divmod(r, 2)
            i, j = divmod(r, n)
            yield i, j, u, k

    @staticmethod
    def _pair_repack(f, plist):
        """Flattens a sequence of pairs to pass through `f`, and then
        re-pairs the result.
        """
        ulist = f(u for p in plist for u in p)
        for u in ulist:
            v = next(ulist)
            yield u, v

    def int_pairs(self, plist):
        """Deprecated alias for `iter_chimera_to_linear_pairs`."""
        msg = ('chimera_coordinates.int_pairs is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.iter_chimera_to_linear_pairs instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_chimera_to_linear_pairs(plist)

    def iter_chimera_to_linear_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of 4-term Chimera
        coordinates to pairs of linear indices.
        """
        return self._pair_repack(self.ints, plist)

    def tuple_pairs(self, plist):
        """Deprecated alias for `iter_linear_to_chimera_pairs`."""
        msg = ('chimera_coordinates.tuple_pairs is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.iter_linear_to_chimera_pairs instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_linear_to_chimera_pairs(plist)

    def iter_linear_to_chimera_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of linear indices
        to pairs of 4-term Chimera coordinates.
        """
        return self._pair_repack(self.tuples, plist)


def linear_to_chimera(r, m, n=None, t=None):
    """Convert the linear index `r` into a chimera index.

    Parameters
    ----------
    r : int
        The linear index value.
    m : int
        Number of rows in the Chimera lattice.
    n : int (optional, default m)
        Number of columns in the Chimera lattice.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.


    Returns
    -------
    i : int
        The column of the Chimera index's unit cell associated with `r`.
    j : int
        The row of the Chimera index's unit cell associated with `r`.
    u : int
        Whether the index is even (0) or odd (1); the side of the bi-partite
        graph of the Chimera unit cell.
    k : int
        Index into the Chimera unit cell.

    Examples
    --------

    >>> G = dnx.linear_to_chimera(212, 8, 8, 4)
    (3, 2, 1, 0)

    """
    return chimera_coordinates(m, n, t).linear_to_chimera(r)


def chimera_to_linear(i, j, u, k, m, n=None, t=None):
    """Convert the chimera index `(i, j, u, k)` into a linear index.

    Parameters
    ----------
    i : int
        The column of the Chimera index's unit cell associated with `r`.
    j : int
        The row of the Chimera index's unit cell associated with `r`.
    u : int
        Whether the index is even (0) or odd (1); the side of the bi-partite
        graph of the Chimera unit cell.
    k : int
        Index into the Chimera unit cell.
    m : int
        Number of rows in the Chimera lattice.
    n : int (optional, default m)
        Number of columns in the Chimera lattice.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.


    Returns
    -------
    r : int
        The linear index node label corresponding to `(i, j, u, k)`.

    Examples
    --------

    >>> G = dnx.chimera_to_linear(3, 2, 1, 0, 8, 8, 4)
    212

    """
    return chimera_coordinates(m, n, t).chimera_to_linear((i, j, u, k))
