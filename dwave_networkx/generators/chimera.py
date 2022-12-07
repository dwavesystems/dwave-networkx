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

"""
Generators for graphs derived from the D-Wave System.

"""
import warnings

import networkx as nx
from networkx.algorithms.bipartite import color
from networkx import diameter

from dwave_networkx.exceptions import DWaveNetworkXException

from itertools import product

from .common import _add_compatible_nodes, _add_compatible_edges, _add_compatible_terms

__all__ = ['chimera_graph',
           'chimera_coordinates',
           'find_chimera_indices',
           'chimera_to_linear',
           'linear_to_chimera',
           'chimera_sublattice_mappings',
           'chimera_torus',
           ]


def chimera_graph(m, n=None, t=None, create_using=None, node_list=None, edge_list=None,
                  data=True, coordinates=False, check_node_list=False, check_edge_list=False):
    """Creates a Chimera lattice of size (m, n, t).

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
        Iterable of nodes in the graph. The nodes should typically be 
        compatible with the requested lattice-shape parameters and coordinate 
        system; incompatible nodes are accepted unless you set :code:`check_node_list=True`. 
        If not specified, calculated from (``m``, ``n``, ``t``) and 
        ``coordinates`` per the topology description below; all :math:`2 t m n`
        nodes are included.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. Edges must be 2-tuples of the nodes 
        specified in ``node_list``, or calculated from (``m``, ``n``, ``t``) and 
        ``coordinates`` per the topology description below; incompatible edges 
        are ignored unless you set :code:`check_edge_list=True`. If not 
        specified, all edges compatible with the ``node_list`` and topology 
        description are included.
    data : bool (optional, default :code:`True`)
        If :code:`True`, each node has a `chimera_index attribute`. The 
        attribute is a 4-tuple Chimera index as defined below.
    coordinates : bool (optional, default :code:`False`)
        If :code:`True`, node labels are 4-tuples, equivalent to the chimera_index
        attribute as below.  In this case, the ``data`` parameter controls the
        existence of a `linear_index attribute`, which is an integer.
    check_node_list : bool (optional, default :code:`False`)
        If :code:`True`, the ``node_list`` elements are checked for compatibility with
        the graph topology and node labeling conventions, and an error is thrown
        if any node is incompatible or duplicates exist. 
        In other words, the ``node_list`` must specify a subgraph of the 
        full-yield graph described below. An exception is allowed if 
        ``check_edge_list=False``, in which case any node in ``edge_list`` is treated as valid.
    check_edge_list : bool (optional, default :code:`False`)
        If :code:`True`, the ``edge_list`` elements are checked for compatibility with
        the graph topology and node labeling conventions, an error is thrown
        if any edge is incompatible or duplicates exist. 
        In other words, the ``edge_list`` must specify a subgraph of the 
        full-yield graph described below.

    Returns
    -------
    G : NetworkX Graph
        An (m, n, t) Chimera lattice. Nodes are labeled by integers.


    A Chimera lattice is an m-by-n grid of Chimera tiles. Each Chimera
    tile is itself a bipartite graph with shores of size t. The
    connection in a Chimera lattice can be expressed using a node-indexing
    notation (i, j, u, k) for each node.

    * (i, j) indexes the (row, column) of the Chimera tile. i must be
      between 0 and m - 1, inclusive, and j must be between 0 and
      n - 1, inclusive.
    * u=0 indicates the left-hand nodes in the tile, and u=1 indicates
      the right-hand nodes.
    * k=0, 1, ..., t - 1 indexes nodes within either the left- or
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
        check_edge_list = False
    if node_list is None:
        check_node_list = False
    
    if edge_list is None or check_edge_list is True:
        if coordinates:
            # tile edges
            G.add_edges_from(((i, j, 0, k0), (i, j, 1, k1))
                             for i in range(m)
                             for j in range(n)
                             for k0 in range(t)
                             for k1 in range(t))

            # horizontal edges
            G.add_edges_from(((i, j, 1, k), (i, j + 1, 1, k))
                             for i in range(m)
                             for j in range(n - 1)
                             for k in range(t))
            # vertical edges
            G.add_edges_from(((i, j, 0, k), (i + 1, j, 0, k))
                             for i in range(m - 1)
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
        if edge_list is not None:
            _add_compatible_edges(G, edge_list)
            
    else:
        if check_node_list or node_list is None:
            if coordinates:
                G.add_nodes_from((i, j, u, k) for i in range(m)
                                  for j in range(n)
                                  for u in range(2)
                                  for k in range(t))
            else:
                G.add_nodes_from(i for i in range(m*n*t*2))
        
        G.add_edges_from(edge_list)

    if node_list is not None:
        if check_node_list:
            _add_compatible_nodes(G, node_list)
        else:
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

    See the :func:`~chimera_graph()` function for a definition of a Chimera graph and Chimera
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
    :func:`.chimera_graph` : Describes the various conventions.

    """
    def __init__(self, m, n=None, t=None):
        self.args = m, m if n is None else n, 4 if t is None else t

    def int(self, q):
        """Deprecated alias for `chimera_to_linear`."""
        msg = ('chimera_coordinates.int is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'chimera_coordinates.chimera_to_linear instead')
        warnings.warn(msg, DeprecationWarning)
        return self.chimera_to_linear(q)

    def chimera_to_linear(self, q):
        """Convert a 4-term Chimera coordinate to a linear index.

        Parameters
        ----------
        q : 4-tuple
            Chimera coordinate.

        Examples
        --------
        >>> dnx.chimera_coordinates(16).chimera_to_linear((2, 2, 0, 0))
        272

        """
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
        """Convert a linear index to a 4-term Chimera coordinate.

        Parameters
        ----------
        r : int
            Linear index.

        Examples
        --------
        >>> dnx.chimera_coordinates(16).linear_to_chimera(272)
        (2, 2, 0, 0)
        """
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
        return self._pair_repack(self.iter_chimera_to_linear, plist)

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
        return self._pair_repack(self.iter_linear_to_chimera, plist)

    def graph_to_linear(self, g):
        """Return a copy of the graph g relabeled to have linear indices"""
        labels = g.graph.get('labels')
        if labels == 'int':
            return g.copy()
        elif labels == 'coordinate':
            return chimera_graph(
                g.graph['rows'],
                n=g.graph['columns'],
                t=g.graph['tile'],
                node_list=self.iter_chimera_to_linear(g),
                edge_list=self.iter_chimera_to_linear_pairs(g.edges),
                data=g.graph['data'],
            )
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.chimera_graph."
            )

    def graph_to_chimera(self, g):
        """Return a copy of the graph g relabeled to have chimera coordinates"""
        labels = g.graph.get('labels')
        if labels == 'int':
            return chimera_graph(
                g.graph['rows'],
                n=g.graph['columns'],
                t=g.graph['tile'],
                node_list=self.iter_linear_to_chimera(g),
                edge_list=self.iter_linear_to_chimera_pairs(g.edges),
                data=g.graph['data'],
                coordinates=True,
            )
        elif labels == 'coordinate':
            return g.copy()
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.chimera_graph."
            )

class __chimera_coordinates_cache_dict(dict):
    """An internal-use cached factory for `chimera_coordinates` objects"""

    def __missing__(self, key):
        self[key] = val = chimera_coordinates(*key)
        return val


_chimera_coordinates_cache = __chimera_coordinates_cache_dict()

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
    return _chimera_coordinates_cache[m, n, t].linear_to_chimera(r)


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
    return _chimera_coordinates_cache[m, n, t].chimera_to_linear((i, j, u, k))


def _chimera_sublattice_mapping(source_to_chimera, chimera_to_target, offset):
    """Constructs a mapping from one chimera graph to another, via an offset.
    This function is used by chimera_sublattice_mappings, and serves to 
    construct a closure that is stable under iteration therein.

    Parameters
    ----------
        source_to_chimera : function
            A function mapping a source node to a chimera-coordinate 
        chimera_to_target: function
            A function mapping a chimera coordinate to a target nodes
        offset : tuple (int, int)
            A pair of ints representing the y- and x-offset of the sublattice

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Chimera
            graph to the target Chimera graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.
        
    """
    y_offset, x_offset = offset

    def mapping(q):
        y, x, u, k = source_to_chimera(q)
        return chimera_to_target((y + y_offset, x + x_offset, u, k))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping


def chimera_sublattice_mappings(source, target, offset_list=None):
    """Yields mappings from a Chimera graph into a larger Chimera graph.
    
    A sublattice mapping is a function from nodes of a
    ``chimera_graph(m_s, n_s, t)`` to nodes of a ``chimera_graph(m_t, n_t, t)``
    with ``m_s <= m_t`` and ``n_s <= n_t``.  This is used to identify subgraphs 
    of the target Chimera graphs which are isomorphic to the source Chimera
    graph.  However, if the target graph is not of perfect yield, these 
    functions do not generally produce isomorphisms (for example, if a node is
    missing in the target graph, it may still appear in the image of the source
    graph).
    
    Note that we do not produce mappings between Chimera graphs of different
    tile parameters, and the mappings produced are not exhaustive.  The mappings
    take the form
    
        ``(y, x, u, k) -> (y + y_offset, x + x_offset, u, k)``
        
    preserving the orientation and tile index of nodes.  We use the notation of
    Chimera coordinates above, but either or both of the target graph may have
    integer or coordinate labels.
    
    Academic note: the full group of isomorphisms of a Chimera graph includes 
    mappings which permute tile indices on a per-row and per-column basis, in
    addition to reflections and rotations of the grid of unit cells where 
    rotations by 90 and 270 degrees induce a change in orientation.  The full
    set of sublattice mappings would take those isomorphisms into account; we do
    not undertake that complexity here.
    
    Parameters
    ----------
        source : NetworkX Graph
            The Chimera graph that nodes are input from
        target : NetworkX Graph
            The Chimera graph that nodes are input from
        offset_list : iterable (tuple), optional (default None)
            An iterable of offsets.  This can be used to reconstruct a set of
            mappings, as the offset used to generate a single mapping is stored
            in the ``offset`` attribute of that mapping.
            
    Yields
    ------
        mapping : function
            A function from nodes of the source graph, to nodes of the target
            graph.  The offset used to generate this mapping is stored in
            ``mapping.offset`` -- these can be collected and passed into 
            ``offset_list`` in a later session.

    """
    if not (source.graph.get('family') == target.graph.get('family') == 'chimera'):
        raise ValueError("source and target graphs must be Chimera graphs constructed by dwave_networkx.chimera_graph")
    
    t = source.graph['tile']
    if t != target.graph['tile']:
        raise ValueError("Cannot construct a sublattice mappings between Chimera graphs with different tile parameters")

    m_s = source.graph['rows']
    n_s = source.graph['columns']
    labels_s = source.graph['labels']
    if labels_s == 'int':
        source_to_chimera = _chimera_coordinates_cache[m_s, n_s, t].linear_to_chimera
    elif labels_s == 'coordinate':
        def source_to_chimera(q):
            return q
    else:
        raise ValueError(f"Chimera node labeling {labels_s} not recognized")

    m_t = target.graph['rows']
    n_t = target.graph['columns']
    labels_t = target.graph['labels']
    if labels_t == 'int':
        chimera_to_target = _chimera_coordinates_cache[m_t, n_t, t].chimera_to_linear
    elif labels_t == 'coordinate':
        def chimera_to_target(q):
            return q
    else:
        raise ValueError(f"Chimera node labeling {labels_t} not recognized")
    
    if offset_list is None:
        y_offsets = range(m_t - m_s + 1)
        x_offsets = range(n_t - n_s + 1)
        offset_list = product(y_offsets, x_offsets)

    for offset in offset_list:
        yield _chimera_sublattice_mapping(source_to_chimera, chimera_to_target, offset)

def chimera_torus(m, n=None, t=None, node_list=None, edge_list=None):
    """Creates a defect-free Chimera lattice of size :math:`(m, n, t)` subject to periodic boundary conditions.


    Parameters
    ----------
    m : int
        Number of rows in the Chimera torus lattice.
        If :math:`m<3` translational invariance already applies in the rows. If 
        :math:`m>=3` additional external couplers are added, reestablishing 
        translational invariance.
        Connectivity of all horizontal qubits is :math:`min(m - 1, 2) + 2t`.
    n : int (optional, default m)
        Number of columns in the Chimera torus lattice.
        If :math:`n<3` translational invariance already applies in the columns. If 
        :math:`n>=3` additional external couplers are added, reestablishing 
        translational invariance.
        Connectivity of all vertical qubits is :math:`min(n - 1, 2) + 2t`.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph. If None, nodes are generated 
        for an undiluted torus calculated from ``m``, ``n`` and ``t``
        as described below. The node list must describe a subset
        of the torus nodes to be maintained in the graph 
        using the coordinate node labeling scheme.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. If None, edges are generated
        for an undiluted torus calculated from ``m``, ``n`` and ``t``
        as described below. The edge list must describe 
        a subgraph of the torus, using the coordinate node labeling scheme.

    Returns
    -------
    G : NetworkX Graph
        A Chimera torus with shape (m, n, t), with Chimera coordinate node labels.


    A Chimera torus is a generalization of the standard chimera graph
    whereby degree-six connectivity is maintained, but the boundary
    condition is modified to enforce an additional translational-invariance 
    symmetry [RH]_. Local connectivity in the Chimera torus
    is identical to connectivity for chimera graph nodes away from the boundary.
    The graph has :code:`V=8*m*n` nodes, and :code:`min(6, 4 + m)V//2 + 
    min(6, 4 + n)V/2` edges. With the standard :math:`K_{t, t}` Chimera tile definition,
    any tile displacement :math:`(x, y)`  modulo :math:`(m, n)`, rows and columns respectively,
    that is, :code:`(i, j, u, k)` -> :code:`((i + x)%m, (i + y)%n, u, k)`,
    defines an automorphism.

    See :func:`.chimera_graph` for additional information.

    Examples
    ========
    >>> G = dnx.chimera_torus(3, 3, 4)  # a 3x3 tile chimera graph (connectivity 6)
    >>> len(G)
    72
    >>> any([len(list(G.neighbors(n))) != 6 for n in G.nodes])
    False

    """
    # Graph properties are by and large inherited from chimera_graph
    G = chimera_graph(m=m, n=n, t=t, node_list=None, edge_list=None, data=True, coordinates=True)
    if n is None:
        n = G.graph['columns']         
    if t is None:
        t = G.graph['tile']
    

    # With modification of the boundary condition
    if m>2:
        # Wrapped around row external-coupler edges:
        additional_edges = [((m - 1, j, 0, k), (0, j, 0, k))
                            for j in range(n)
                            for k in range(t)]
    else:
        additional_edges = []
    
    if n>2:
        # Wrapped around columns external-coupler edges:
        additional_edges += [((i, n - 1, 1, k), (i, 0, 1, k))
                             for i in range(m)
                             for k in range(t)]

    if len(additional_edges)>0:
        G.add_edges_from(additional_edges)

    _add_compatible_terms(G, node_list, edge_list)

    G.graph['boundary_condition'] = 'torus'
    
    return G
