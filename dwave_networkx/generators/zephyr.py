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


__all__ = ['zephyr_graph',
           'zephyr_coordinates',
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
        v = 0
        def coord_label():
            return q
        def int_label():
            return v
        if coordinates:
            other_name = 'linear_index'
            this_label = coord_label
            other_label = int_label
        else:
            other_name = 'zephyr_index'
            this_label = int_label
            other_label = coord_label
        for u in range(2):
            for w in range(M):
                for k in range(t):
                    for j in (0, 1):
                        for z in range(m):
                            q = u, w, k, j, z
                            p = this_label()
                            if p in G:
                                G.nodes[p][other_name] = other_label()
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
        self.args = m, 2*m+1, t

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
