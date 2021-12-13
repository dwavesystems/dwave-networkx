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

__all__ = ['zephyr_graph',
           'zephyr_coordinates',
           'zephyr_sublattice_mappings',
           ]


def zephyr_graph(m, t=4, create_using=None, node_list=None, edge_list=None,
                   data=True, coordinates=False):
    """
    Creates a Zephyr graph [brk]_ with grid parameter ``m`` and tile parameter ``t``.

    Parameters
    ----------
    m : int
        Grid parameter for the Zephyr lattice.
    t : int
        Tile parameter for the Zephyr lattice.
    create_using : Graph, optional (default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable, optional (default None)
        Iterable of nodes in the graph. If None, calculated from ``m``.
        Note that this list is used to remove nodes, so only specified nodes
        that belong to the base node set (described in the ``coordinates``
        parameter below) will be added.
    edge_list : iterable, optional (default None)
        Iterable of edges in the graph. If None, edges are generated as
        described below. The nodes in each edge must be labeled according to the
        ``coordinates`` parameter, described below.
    data : bool, optional (default True)
        If True, adds to each node an attribute with a format that depends on
        the ``coordinates`` parameter:
            a 5-tuple ``'zephyr_index'`` if ``coordinates`` is False
            an integer ``'linear_index'`` if ``coordinates`` is True
    coordinates : bool, optional (default False)
        If True, node labels are 5-tuple Zephyr indices.

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

        ``I(0, w, k, j, z) = [(2*z+j, w, 0, 2*k+j), (2*z+1+j, w, 0, 2*k+j)]``
        ``I(1, w, k, j, z) = [(w, 2*z+j, 1, 2*k+j), (w, 2*z+1+j, 1, 2*k+j)]``

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

        ``q = (((u * (2 * m + 1) + w) * t + k) * 2 + j) * m + z``


    Examples
    --------
    >>> G = dnx.zephyr_graph(2)
    >>> G.nodes(data=True)[(0, 0, 0, 0, 0)]    # doctest: +SKIP
    {'linear_index': 0}

    References
    ----------
    .. [brk] Boothby, Raymond, King, Zephyr Topology of D-Wave Quantum
        Processors, October 2021.
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
        #external edges
        G.add_edges_from((label(u, w, k, j, z), label(u, w, k, j, z + 1))
                         for u, w, k, j, z in product(
                            (0, 1), range(M), range(t), (0, 1), range(m-1)
                         ))

        #odd edges
        G.add_edges_from((label(u, w, k, 0, z), label(u, w, k, 1, z-a))
                         for u, w, k, a in product(
                            (0, 1), range(M), range(t), (0, 1)
                         )
                         for z in range(a, m))

        #internal edges
        G.add_edges_from((label(0, 2*w+1+a*(2*i-1), k, j, z), label(1, 2*z+1+b*(2*j-1), h, i, w))
                         for w, z, h, k, i, j, a, b in product(
                            range(m), range(m), range(t), range(t), (0, 1), (0, 1), (0, 1), (0, 1)
                         ))

    else:
        G.add_edges_from(edge_list)

    if node_list is not None:
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
        """Convert a 5-term Zephyr coordinate into a linear index.

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
        """Convert a linear index into a 5-term Zephyr coordinate.

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
        """Return an iterator converting a sequence of 5-term Zephyr
        coordinates to linear indices.
        """
        m, M, t = self.args
        for (u, w, k, j, z) in qlist:
            yield (((u * M + w) * t + k) * 2 + j) * m + z

    def iter_linear_to_zephyr(self, rlist):
        """Return an iterator converting a sequence of linear indices to 5-term
        Zephyr coordinates.
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
        """Flattens a sequence of pairs to pass through `f`, and then
        re-pairs the result.
        """
        ulist = f(u for p in plist for u in p)
        for u in ulist:
            v = next(ulist)
            yield u, v

    def iter_zephyr_to_linear_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of 5-term Zephyr
        coordinates to pairs of linear indices.
        """
        return self._pair_repack(self.iter_zephyr_to_linear, plist)

    def iter_linear_to_zephyr_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of linear indices
        to pairs of 5-term Zephyr coordinates.
        """
        return self._pair_repack(self.iter_linear_to_zephyr, plist)

    def graph_to_linear(self, g):
        """Return a copy of the graph g relabeled to have linear indices"""
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
        """Return a copy of the graph g relabeled to have zephyr coordinates"""
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

    #store the offset in the mapping, so the user can reconstruct it
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

    #store the offset in the mapping, so the user can reconstruct it
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
            
    #store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping


def zephyr_sublattice_mappings(source, target, offset_list=None):
    """Yields mappings from a Chimera or Zephyr graph into a Zephyr graph.
    
    A sublattice mapping is a function from nodes of
        * a ``zephyr_graph(m_s, t)`` to nodes of a ``zephyr_graph(m_t, t)``
            where ``m_s <= m_t``,
        * a ``chimera_graph(m_s, n_s, t)`` to nodes of a ``zephyr_graph(m_t, t)``
            where ``m_s <= 2*m_t`` and ``n_s <= 2*m_t``, or
        * a ``chimera_graph(m_s, n_s, 2*t)`` to nodes of a ``zephyr_graph(m_t, t)``
            where ``m_s <= m_t`` and ``n_s <= m_t``, or

    This is used to identify subgraphs of the target Zephyr graphs which are 
    isomorphic to the source graph.  However, if the target graph is not of
    perfect yield, these functions do not generally produce isomorphisms (for
    example, if a node is missing in the target graph, it may still appear in
    the image of the source graph).
    
    Note that we require the tile parameter of Chimera graphs to be either the
    same our double that of the target Zephyr graphs; or if both graphs are
    Zephyr graphs, we require the tile parameters to be the same.  The mappings
    we produce preserve the linear ordering of tile indices; see 
    ``_zephyr_zephyr_sublattice_mapping``,
    ``_double_chimera_zephyr_sublattice_mapping``, and
    ``_single_chimera_zephyr_sublattice_mapping`` for more details.
    
    Academic note: the full group of isomorphisms of a Chimera graph includes 
    mappings which permute tile indices on a per-row and per-column basis, in
    addition to reflections and rotations of the grid of unit cells where 
    rotations by 90 and 270 degrees induce a change in orientation.  The
    isomorphisms of Zephyr graphs permit permutations of major tile indices on a 
    per-row and per-column basis, in addition to reflections of the grid which
    induce inversion of orthogonal minor offsets, and rotations which induce 
    inversions of minor offsets and/or orientation. The full set of sublattice
    mappings would take those isomorphisms into account; we do not undertake
    that complexity here.

    Parameters
    ----------
        source : NetworkX Graph
            The Chimera or Zephyr graph that nodes are input from
        target : NetworkX Graph
            The Zephyr graph that nodes are output to
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
    if target.graph.get('family') != 'zephyr':
        raise ValueError("source graphs must a Zephyr graph constructed by dwave_networkx.zephyr_graph")

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

