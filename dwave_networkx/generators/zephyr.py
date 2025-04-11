# Copyright 2021 D-Wave Systems Inc.
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
Generators for some graphs derived from the D-Wave System.
"""
import re
import warnings
from itertools import product

import networkx as nx

from dwave_networkx.exceptions import DWaveNetworkXException

from .chimera import _chimera_coordinates_cache

from .common import _add_compatible_edges, _add_compatible_nodes, _add_compatible_terms

__all__ = ['zephyr_graph',
           'zephyr_coordinates',
           'zephyr_sublattice_mappings',
           'zephyr_torus',
           'zephyr_four_color',
           ]

def zephyr_graph(m, t=4, create_using=None, node_list=None, edge_list=None,
                 data=True, coordinates=False, check_node_list=False,
                 check_edge_list=False):
    """
    Creates a Zephyr graph with grid parameter ``m`` and tile parameter ``t``.

    The Zephyr topology is described in [Boo2021]_.

    Parameters
    ----------
    m : int
        Grid parameter for the Zephyr lattice.
    t : int
        Tile parameter for the Zephyr lattice.
    create_using : Graph, optional (default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph. If not specified, calculated from (``m``, ``t``)
        and ``coordinates``. The nodes should typically be compatible with the 
        requested lattice shape parameters and coordinate system, incompatible 
        nodes are accepted unless you set :code:`check_node_list=True`. If not 
        specified, all :math:`4 t m (2 m + 1)` nodes compatible with the 
        topology description are included.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. Edges must be 2-tuples of the nodes 
        specified in node_list, or calculated from (``m``, ``t``) and ``coordinates`` 
        per the topology description below; incompatible edges are ignored 
        unless you set :code:`check_edge_list=True`. If not specified, all edges
        compatible with the ``node_list`` and topology description are included.
    data : bool, optional (default :code:`True`)
        If :code:`True`, adds to each node an attribute with a format that depends on
        the ``coordinates`` parameter: a 5-tuple ``'zephyr_index'`` if
        :code:`coordinates=False` and an integer ``'linear_index'`` if ``coordinates``
        is :code:`True`.
    coordinates : bool, optional (default :code:`False`)
        If :code:`True`, node labels are 5-tuple Zephyr indices.
    check_node_list : bool (optional, default :code:`False`)
        If :code:`True`, the ``node_list`` elements are checked for compatibility with
        the graph topology and node labeling conventions, and an error is thrown
        if any node is incompatible or duplicates exist. 
        In other words, ``node_lists`` must specify a subgraph of the default 
        (full yield) graph described below. An exception is allowed if 
        ``check_edge_list=False``, any node in edge_list will also be treated as valid.
    check_edge_list : bool (optional, default :code:`False`)
        If :code:`True`, ``edge_list`` elements are checked for compatibility with
        the graph topology and node labeling conventions, and an error is thrown
        if any edge is incompatible or duplicates exist. 
        In other words, ``edge_list`` must specify a subgraph of the default 
        (full yield) graph described below.

    Returns
    -------
    G : NetworkX Graph
        A Zephyr lattice for grid parameter ``m`` and tile parameter ``t``.


    The maximum degree of this graph is :math:`4t+4`. The number of nodes is
    given by

        * ``zephyr_graph(m, t)``: :math:`4tm(2m+1)`

    The number of edges depends on parameter settings,

        * ``zephyr_graph(1, t)``: :math:`2t(8t+3)`
        * ``zephyr_graph(m, t)``: :math:`2t((8t+8)m^2-2m-3)`  if m > 1

    A Zephyr lattice is a graph minor of a lattice similar to Chimera, where
    unit tiles have odd couplers similar to Pegasus graphs. In its most
    general definition, prelattice :math:`Q(2m+1)` contains nodes of the form

        * vertical nodes: :math:`(i, j, 0, k)` with :math:`0 <= k < 2t`
        * horizontal nodes: :math:`(i, j, 1, k)` with :math:`0 <= k < 2t`

    for :math:`0 <= i < 2m+1` and :math:`0 <= j < 2m+1`, and edges of the form

        * external: :math:`(i, j, u, k)` ~ :math:`(i+u, j+1-u, u, k)`
        * internal: :math:`(i, j, 0, k)` ~ :math:`(i, j, 1, h)`
        * odd: :math:`(i, j, u, 2k)` ~ :math:`(i, j, u, 2k+1)`

    The minor---a Zephyr lattice---is constructed by contracting pairs of
    external edges::

        I(0, w, k, j, z) = [(2*z+j, w, 0, 2*k+j), (2*z+1+j, w, 0, 2*k+j)]
        I(1, w, k, j, z) = [(w, 2*z+j, 1, 2*k+j), (w, 2*z+1+j, 1, 2*k+j)]

    and deleting the prelattice nodes of any pair not fully contained in
    :math:`Q(2m+1)`.

    The *Zephyr index* of a node in a Zephyr lattice, :math:`(u, w, k, j, z)`,
    can be interpreted as:

        * :math:`u`: qubit orientation (0 = vertical, 1 = horizontal)
        * :math:`w`: orthogonal major offset; :math:`0 <= w < 2m+1`
        * :math:`k`: orthogonal secondary offset; :math:`0 <= k < t`
        * :math:`j`: orthogonal minor offset; :math:`0 <= j < 2`
        * :math:`z`: parallel offset; :math:`0 <= z < m`

    Edges in the minor have the form

        * external: :math:`(u, w, k, j, z)` ~ :math:`(u, w, k, j, z+1)`
        * odd: :math:`(u, w, 2k, z)` ~ :math:`(u, w, 2k+1, z-a)`
        * internal: :math:`(0, 2w+1-a, k, j, z-jb)` ~ :math:`(1, 2z+1-b, h, i, w-ia)`

    for :math:`0 <= a < 2` and :math:`0 <= b < 2`, where internal edges only exist when

        1. :math:`0 <= 2w+1-a < 2m+1`,
        2. :math:`0 <= 2z+1-a < 2m+1`,
        3. :math:`0 <= z-jb < m`, and
        4. :math:`0 <= w-ia < m`.

    Linear indices are computed from Zephyr indices by the formula::

        q = (((u * (2 * m + 1) + w) * t + k) * 2 + j) * m + z


    Examples
    --------
    >>> G = dnx.zephyr_graph(2)
    >>> G.nodes(data=True)[(0, 0, 0, 0, 0)]    # doctest: +SKIP
    {'linear_index': 0}

    References
    ----------
    Boothby, Raymond, King. 
    Zephyr Topology of D-Wave Quantum Processors 
    October 2021.
    https://dwavesys.com/media/fawfas04/14-1056a-a_zephyr_topology_of_d-wave_quantum_processors.pdf
    """
    G = nx.empty_graph(0, create_using)
    m = int(m)
    t = int(t)

    G.name = "zephyr_graph(%s, %s)" % (m, t)

    M = 2*m+1

    if coordinates:
        def label(*q):
            return q
        labels = 'coordinate'
    else:
        labels = 'int'
        def label(u, w, k, j, z):
            return (((u * M + w) * t + k) * 2 + j) * m + z

    construction = (("family", "zephyr"), ("rows", m), ("columns", m),
                    ("tile", t), ("data", data), ("labels", labels))

    G.graph.update(construction)
    
    if edge_list is None:
        check_edge_list = False
    if node_list is None:
        check_node_list = False
    
    if edge_list is None or check_edge_list is True:
        # external edges
        G.add_edges_from((label(u, w, k, j, z), label(u, w, k, j, z + 1))
                         for u, w, k, j, z in product(
                            (0, 1), range(M), range(t), (0, 1), range(m-1)
                         ))

        # odd edges
        G.add_edges_from((label(u, w, k, 0, z), label(u, w, k, 1, z-a))
                         for u, w, k, a in product(
                            (0, 1), range(M), range(t), (0, 1)
                         )
                         for z in range(a, m))

        # internal edges
        G.add_edges_from((label(0, 2*w+1+a*(2*i-1), k, j, z), label(1, 2*z+1+b*(2*j-1), h, i, w))
                         for w, z, h, k, i, j, a, b in product(
                            range(m), range(m), range(t), range(t), (0, 1), (0, 1), (0, 1), (0, 1)
                         ))
        if edge_list is not None:
            _add_compatible_edges(G, edge_list)
    else:
        if check_node_list or node_list is None:
            G.add_nodes_from(label(u, w, k, j, z) for u in range(2)
                             for w in range(2*m+1)
                             for k in range(t)
                             for j in range(2)
                             for z in range(m))
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
            def fill_data():
                d = get_node_data((u, w, k, j, z))
                if d is not None:
                    d['linear_index'] = v

        else:
            def fill_data():
                d = get_node_data(v)
                if d is not None:
                    d['zephyr_index'] = (u, w, k, j, z)

        v = 0
        get_node_data = G.nodes.get
        for u in range(2):
            for w in range(M):
                for k in range(t):
                    for j in (0, 1):
                        for z in range(m):
                            fill_data()
                            v += 1

    return G


# Developer note: we could implement a function that creates the iter_*_to_* and
# iter_*_to_*_pairs methods just-in-time, but there are a small enough number
# that for now it makes sense to do them by hand.
class zephyr_coordinates(object):
    """Provides coordinate converters for the Zephyr indexing schemes.

    Parameters
    ----------
    m : int
        Grid parameter for the Zephyr lattice.
    t : int
        Tile parameter for the Zephyr lattice; must be even.

    See also
    --------
    :func:`.zephyr_graph` : Describes the various coordinate conventions.

    """
    def __init__(self, m, t=4):
        self.args = m, 2 * m + 1, t

    def zephyr_to_linear(self, q):
        """Converts a 5-term Zephyr coordinate into a linear index.

        Parameters
        ----------
        q : 5-tuple
            Zephyr coordinate.

        Examples
        --------
        >>> dnx.zephyr_coordinates(2).zephyr_to_linear((0, 1, 2, 1, 0))
        26
        """
        u, w, k, j, z = q
        m, M, t = self.args
        return (((u * M + w) * t + k) * 2 + j) * m + z

    def linear_to_zephyr(self, r):
        """Converts a linear index into a 5-term Zephyr coordinate.

        Parameters
        ----------
        r : int
            Linear index.

        Examples
        --------
        >>> dnx.zephyr_coordinates(2).linear_to_zephyr(137)
        (1, 3, 2, 0, 1)

        """
        m, M, t = self.args
        r, z = divmod(r, m)
        r, j = divmod(r, 2)
        r, k = divmod(r, t)
        u, w = divmod(r, M)
        return u, w, k, j, z

    def iter_zephyr_to_linear(self, qlist):
        """Converts a sequence of 5-term Zephyr coordinates to linear indices.
        """
        m, M, t = self.args
        for (u, w, k, j, z) in qlist:
            yield (((u * M + w) * t + k) * 2 + j) * m + z

    def iter_linear_to_zephyr(self, rlist):
        """Converts a sequence of linear indices to 5-term Zephyr coordinates.
        """
        m, M, t = self.args
        for r in rlist:
            r, z = divmod(r, m)
            r, j = divmod(r, 2)
            r, k = divmod(r, t)
            u, w = divmod(r, M)
            yield u, w, k, j, z

    @staticmethod
    def _pair_repack(f, plist):
        """Flattens a sequence of pairs to pass through ``f``, and then
        re-pairs the result.
        """
        ulist = f(u for p in plist for u in p)
        for u in ulist:
            v = next(ulist)
            yield u, v

    def iter_zephyr_to_linear_pairs(self, plist):
        """Converts pairs of 5-term Zephyr coordinates to pairs of linear indices.
        """
        return self._pair_repack(self.iter_zephyr_to_linear, plist)

    def iter_linear_to_zephyr_pairs(self, plist):
        """Converts pairs of linear indices to pairs of 5-term Zephyr coordinates.
        """
        return self._pair_repack(self.iter_linear_to_zephyr, plist)

    def graph_to_linear(self, g):
        """Returns a copy of the graph ``g`` relabeled to have linear indices.
        
        Parameters
        ----------
        g : NetworkX Graph
            The Zephyr graph to be relabeled.        
        
        Returns
        -------
        G : NetworkX Graph
            A Zephyr graph relabeled with linear indices.
        """
        labels = g.graph.get('labels')
        if labels == 'int':
            return g.copy()
        elif labels == 'coordinate':
            nodes = self.iter_zephyr_to_linear(g)
            edges = self.iter_zephyr_to_linear_pairs(g.edges)
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.zephyr_graph."
            )

        return zephyr_graph(
            g.graph['rows'],
            t = g.graph['tile'],
            node_list=nodes,
            edge_list=edges,
            data=g.graph['data'],
        )

    def graph_to_zephyr(self, g):
        """Returns a copy of the graph ``g`` relabeled to have Zephyr coordinates.
        
        Parameters
        ----------
        g : NetworkX Graph
            The Zephyr graph to be relabeled.        
        
        Returns
        -------
        G : NetworkX Graph
            A Zephyr graph relabeled with Zephyr coordinates.
        """
        labels = g.graph.get('labels')
        if labels == 'int':
            nodes = self.iter_linear_to_zephyr(g)
            edges = self.iter_linear_to_zephyr_pairs(g.edges)
        elif labels == 'coordinate':
            return g.copy()
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.zephyr_graph."
            )

        return zephyr_graph(
            g.graph['rows'],
            t=g.graph['tile'],
            node_list=nodes,
            edge_list=edges,
            data=g.graph['data'],
            coordinates=True,
        )


class __zephyr_coordinates_cache_dict(dict):
    """An internal-use cached factory for `zephyr_coordinates` objects"""

    def __missing__(self, key):
        self[key] = val = zephyr_coordinates(*key)
        return val


_zephyr_coordinates_cache = __zephyr_coordinates_cache_dict()


def _zephyr_zephyr_sublattice_mapping(source_to_zephyr, zephyr_to_target, offset):
    """Constructs a mapping from a Zephyr graph to a Zephyr graph, via an offset.
    This function is used by zephyr_sublattice_mappings, and serves to construct
    a closure that is stable under iteration therein.

    The mappings implemented by this function interpret offsets in the grid of
    the Chimera(2m+1, 2m+1, 2*t) graphs underlying the source and tartget Zephyr
    graphs.  The formulas (see implementation) are somewhat complex, because

        * a shift by a y-unit induces a reversal of the orthogonal minor offset
            (j index) of vertical qubits,
        * a shift by an x-unit induces a reversal of the j index of horizontal
            qubits, and
        * a shift by a unit parallel to a qubit is equivalent to 1/2-unit shift
            in the z-direction (but z-coordinates are integral) which is
            mediated by the j index.

    Parameters
    ----------
        source_to_zephyr : function
            A function mapping a source node to a zephyr coordinate
        zephyr_to_target: function
            A function mapping a zephyr coordinate to a target node
        offset : tuple (int, int)
            A pair of ints representing the y- and x-offset of the sublattice

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Zephyr
            graph to the target Zephyr graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.

    """
    y_offset, x_offset = offset

    delta = [
        [y_offset % 2, x_offset, y_offset],
        [x_offset % 2, y_offset, x_offset],
    ]

    def mapping(q):
        u, w, k, j, z = source_to_zephyr(q)
        dj, dw, dz = delta[u]
        return zephyr_to_target((u, w + dw, k, j ^ dj, z + (dz + j) // 2))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping

def _single_chimera_zephyr_sublattice_mapping(source_to_chimera, zephyr_to_target, offset):
    """Constructs a mapping from a Chimera graph to a Zephyr graph, via an offset.
    This function is used by zephyr_sublattice_mappings, and serves to construct
    a closure that is stable under iteration therein.

    The mappings implemented by this function view a ``chimera(2*m, 2*m, t)`` as
    a subgraph of ``zephyr_graph(m, t)`` through the mapping

        (2*y+j, x, 0, k) -> (0, x, k, j, y)
        (y, 2*x+j, 1, k) -> (1, y, k, j, x)

    which interprets odd couplers of Zephyr as external couplers of Chimera.
    The above is a slight simplification of matters; it is the simplest of a
    family of :math:`(t+1)^2` offsets (see how ``k_offset0`` and ``k_offset``
    are used in the implementation).

    Additionally, the sublattice represented  by the source graph can have x-
    and y-offsets into the chimera graph above, as with ordinary Chimera
    subgraph mappings.

    Parameters
    ----------
        source_to_chimera : function
            A function mapping a source node to a chimera coordinate
        zephyr_to_target: function
            A function mapping a zephyr coordinate to a target node
        offset : tuple (int, int, int, int, int)
            A tuple of ints (t, k_offset0, k_offset1, y_offset, x_offset)
            defining the sublattice mapping

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Zephyr
            graph to the target Zephyr graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.

    """
    t, y_offset, x_offset, k_offset0, k_offset1 = offset

    def mapping(q):
        y, x, u, k = source_to_chimera(q)
        if u:
            dw, k = divmod(k + k_offset1, t)
            z, j = divmod(x + x_offset, 2)
            return zephyr_to_target((u, y + y_offset + dw, k, j, z))
        else:
            dw, k = divmod(k + k_offset0, t)
            z, j = divmod(y + y_offset, 2)
            return zephyr_to_target((u, x + x_offset + dw, k, j, z))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping

def _double_chimera_zephyr_sublattice_mapping(source_to_chimera, zephyr_to_target, offset):
    """Constructs a mapping from a Chimera graph to a Zephyr graph, via an offset.
    This function is used by zephyr_sublattice_mappings, and serves to construct
    a closure that is stable under iteration therein.

    The mappings implemented by this function view a ``chimera(m, m, 2*t)`` as
    a subgraph of ``zephyr_graph(m, t)`` through the mappings

        (y, x, 0, k) -> (0, x, k, j0, y)
        (y, x, 1, k) -> (1, y, k, j1, x)

    where j0 and j1 are each 0 or 1.  Additionally, the sublattice represented
    by the source graph can have x- and y-offsets into the chimera graph above,
    as with ordinary Chimera subgraph mappings.

    Parameters
    ----------
        source_to_chimera : function
            A function mapping a source node to a chimera coordinate
        zephyr_to_target: function
            A function mapping a zephyr coordinate to a target node
        offset : tuple (int, int, int, int, int)
            A tuple of ints (t, j0, j1, y_offset, x_offset) defining the
            sublattice mapping

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Zephyr
            graph to the target Zephyr graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.

    """
    t, y_offset, x_offset, j0, j1 = offset
    def mapping(q):
        y, x, u, k = source_to_chimera(q)
        wz, kz = divmod(k, t)
        if u:
            return zephyr_to_target((u, 2 * (y + y_offset) + j0 + wz, kz, j1, x + x_offset))
        else:
            return zephyr_to_target((u, 2 * (x + x_offset) + j1 + wz, kz, j0, y + y_offset))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping


def zephyr_sublattice_mappings(source, target, offset_list=None):
    r"""Yields mappings from a Chimera or Zephyr graph into a Zephyr graph.

    A sublattice mapping is a function from nodes of

        * a ``zephyr_graph(m_s, t)`` to nodes of a ``zephyr_graph(m_t, t)``
          where ``m_s <= m_t``,
        * a ``chimera_graph(m_s, n_s, t)`` to nodes of a ``zephyr_graph(m_t, t)``
          where ``m_s <= 2*m_t`` and ``n_s <= 2*m_t``, or
        * a ``chimera_graph(m_s, n_s, 2*t)`` to nodes of a ``zephyr_graph(m_t, t)``
          where ``m_s <= m_t`` and ``n_s <= m_t``.

    This sublattice mapping is used to identify subgraphs of the target Zephyr graph 
    which are isomorphic to the source graph. However, if the target graph is not of
    perfect yield,\ [#]_ this function does not generally produce isomorphisms; for
    example, if a node is missing in the target graph, it may still appear in
    the image of the source graph.

    The tile parameter of Chimera graphs must be either the same or double 
    that of the target Zephyr graphs; if both graphs are Zephyr graphs, 
    the tile parameters must be the same. The mappings produced preserve 
    the linear ordering of tile indices; see the
    ``_zephyr_zephyr_sublattice_mapping``,
    ``_double_chimera_zephyr_sublattice_mapping``, and
    ``_single_chimera_zephyr_sublattice_mapping`` internal functions in the source code.
    
    .. [#]
        The yield is the percentage of working qubits on a QPU and the subset 
        of available qubits is called the :ref:`working graph <qpu_topologies>`.
        
    Parameters
    ----------
        source : NetworkX Graph
            The Chimera or Zephyr graph that nodes are input from.
        target : NetworkX Graph
            The Zephyr graph that nodes are output to.
        offset_list : iterable (tuple), optional (default None)
            An iterable of offsets that can be used to reconstruct a set of
            mappings. The offset used to generate a single mapping is stored
            in the ``offset`` attribute of that mapping.

    Yields
    ------
        mapping : function
            A function from nodes of the source graph to nodes of the target
            graph.  The offset used to generate this mapping is stored in
            ``mapping.offset``, which can be collected and passed into
            ``offset_list`` in a later session.

    Notes
    -----
    The full group of isomorphisms of a Chimera graph includes
    mappings which permute tile indices on a per-row and per-column basis in
    addition to reflections and rotations of the grid of unit tiles where
    rotations by 90 and 270 degrees induce a change in orientation.  The
    isomorphisms of Zephyr graphs permit permutations of major tile indices on a
    per-row and per-column basis in addition to reflections of the grid that
    induce inversion of orthogonal minor offsets and rotations that induce
    inversions of minor offsets, orientation, or both. Although the full set 
    of sublattice mappings would take those isomorphisms into account,
    this function does not handle that complex task.
    """
    if target.graph.get('family') != 'zephyr':
        raise ValueError("source graphs must be a Zephyr graph constructed by dwave_networkx.zephyr_graph")

    m_t = target.graph['rows']
    t = target.graph['tile']
    labels_t = target.graph['labels']
    if labels_t == 'int':
        zephyr_to_target = _zephyr_coordinates_cache[m_t, t].zephyr_to_linear
    elif labels_t == 'coordinate':
        def zephyr_to_target(q):
            return q
    else:
        raise ValueError(f"Zephyr node labeling {labels_t} not recognized")

    labels_s = source.graph['labels']
    if source.graph.get('family') == 'chimera':
        t_t = source.graph['tile']
        m_s = source.graph['rows']
        n_s = source.graph['columns']

        if t_t == t:
            make_mapping = _single_chimera_zephyr_sublattice_mapping
            if offset_list is None:
                krange = range(t+1)
                mrange = range(2*m_t - m_s + 1)
                nrange = range(2*m_t - n_s + 1)
                offset_list = product([t], mrange, nrange, krange, krange)
        elif t_t == 2*t:
            make_mapping = _double_chimera_zephyr_sublattice_mapping
            if offset_list is None:
                jrange = range(2)
                mrange = range(m_t - m_s + 1)
                nrange = range(m_t - n_s + 1)
                offset_list = product([t], mrange, nrange, jrange, jrange)
        else:
            raise ValueError(f"Cannot construct sublattice mappings from Chimera to this Zephyr graph unless the tile parameter of the chimera graph is {t} or {2*t}.")

        if labels_s == 'coordinate':
            def source_to_inner(q):
                return q
        elif labels_s == 'int':
            source_to_inner = _chimera_coordinates_cache[m_s, n_s, t_t].linear_to_chimera
        else:
            raise ValueError(f"Chimera node labeling {labels_s} not recognized")

    elif source.graph.get('family') == 'zephyr':
        m_s = source.graph['rows']
        if offset_list is None:
            mrange = range((2*m_t+1) - (2*m_s+1) + 1)
            offset_list = product(mrange, mrange)

        labels_s = source.graph['labels']
        if labels_s == 'int':
            source_to_inner = _zephyr_coordinates_cache[m_s, t].linear_to_zephyr
        elif labels_s == 'coordinate':
            def source_to_inner(q):
                return q
        else:
            raise ValueError(f"Zephyr node labeling {labels_s} not recognized")

        make_mapping = _zephyr_zephyr_sublattice_mapping

    else:
        raise ValueError("source graph must be a Chimera graph or Zephyr graph constructed by dwave_networkx.chimera_graph or dwave_networkx.zephyr_graph respectively")

    for offset in offset_list:
        yield make_mapping(source_to_inner, zephyr_to_target, offset)

def zephyr_torus(m, t=4, node_list=None, edge_list=None):
    """
    Creates a Zephyr graph modified to allow for periodic boundary conditions and translational invariance.
    
    The graph matches the local connectivity properties of a standard Zephyr graph,
    but with modified periodic boundary condition. Tiles of :math:`8t` nodes are arranged
    on an :math:`m` by :math:`m` torus. 

    Parameters
    ----------
    m : int
        Grid parameter for the Zephyr lattice.
        Connectivity of all nodes is :math:`4t + min(2m - 1, 4)`.
    t : int
        Tile parameter for the Zephyr lattice.
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph. If None, nodes are generated 
        for an undiluted torus calculated from ``m`` and ``t``
        as described below. The node list must describe a subset
        of the torus nodes to be maintained in the graph 
        using the coordinate node labeling scheme.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. If None, edges are generated
        for an undiluted torus calculated from ``m`` and ``t``
        as described below. The edge list must describe 
        a subgraph of the torus, using the coordinate node labeling scheme.

    Returns
    -------
    G : NetworkX Graph
        A Zephyr torus with grid parameter ``m`` and tile parameter ``t``,
        with Zephyr coordinate node labels.


    A Zephyr torus is a generalization of the standard Zephyr graph
    whereby degree-twenty connectivity is maintained, but the boundary
    condition is modified to enforce an additional translational-invariance 
    symmetry [Ray2023]_. Local connectivity in the Zephyr torus
    is identical to connectivity for Zephyr graph nodes away from the boundary.
    A tile consists of :math:`8t` nodes, and the torus has :math:`m` by :math:`m` tiles. 
    Tile displacement modulo :math:`m` defines an automorphism.
    
    See :func:`.zephyr_graph` for additional information.

    Examples
    --------
    >>> G = dnx.zephyr_torus(3)  # a 3x3 tile pegasus torus (connectivity 15)
    >>> len(G) # 3*3*24
    288
    >>> any([len(list(G.neighbors(n))) != 20 for n in G.nodes])
    False

    """
    G = zephyr_graph(m=m, t=t, node_list=None, edge_list=None,
                         data=True, coordinates=True)
    
    def relabel(u, w, k, j, z):
        return (u, w%(2*m), k, j, z)
    
    # Contract internal couplers spanning the boundary:
    G.add_edges_from([(relabel(*edge[0]), relabel(*edge[1]))
                      for edge in G.edges() if edge[0][1]==2*m or edge[1][1]==2*m])
    
    if m>1:
        # Add boundary spanning external couplers:
        G.add_edges_from([((u, w, k, 1, m - 1), (u, w, k, 0, 0))
                          for u in range(2)
                          for w in range(2*m)
                          for k in range(t)])
        G.add_edges_from([((u, w, k, j, m - 1), (u, w, k, j, 0))
                          for u in range(2)
                          for w in range(2*m)
                          for k in range(t)
                          for j in range(2)])
        
    # Delete variables contracted at the boundary:
    G.remove_nodes_from([(u, 2*m, k, j, z)
                         for u in range(2) for k in range(t) for j in range(2) for z in range(m)])
    
    _add_compatible_terms(G, node_list, edge_list)
    
    G.graph['boundary_condition'] = 'torus'

    return G


def zephyr_four_color(q, scheme=0):
    """Node color assignment sufficient for four coloring of a Zephyr graph.

    Parameters
    ----------
        q : tuple
            Qubit label in standard coordinate format: u, w, k, j, z
        scheme : int
            Two patterns not related by automorphism are supported 
    Returns
    -------
        color : int
            Colors 0, 1, 2 or 3
    Examples
    ========
    A mapping of every qubit (default integer labels) in the Zephyr[m, t]
    graph to one of 4 colors
    >>> m = 2
    >>> G = dnx.zephyr_graph(m, coordinates=True)
    >>> colors = {q: dnx.zephyr_four_color(q) for q in G.nodes()}
    """
    u, w, _, j, z = q
    
    if scheme == 0:
        return j + ((w + 2*(z+u) + j)&2)
    elif scheme == 1:
        return (2*u + w + 2*z + j) & 3
    else:
        raise ValueError('Unknown scheme')
