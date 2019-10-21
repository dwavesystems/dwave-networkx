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
Generators for some graphs derived from the D-Wave System.

"""
import re

import networkx as nx

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException
import warnings

__all__ = ['pegasus_graph',
           'pegasus_coordinates',
           ]

# compatibility for python 2/3
if _PY2:
    range = xrange


def pegasus_graph(m, create_using=None, node_list=None, edge_list=None, data=True,
                  offset_lists=None, offsets_index=None, coordinates=False, fabric_only=True,
                  nice_coordinates=False):
    """
    Creates a Pegasus graph with size parameter m.  The number of nodes and edges varies
    according to multiple parameters, for example,

        pegasus_graph(1) contains zero nodes,
        pegasus_graph(m, fabric_only=False) contains :math:`24m(m-1)` nodes,
        pegasus_graph(m, fabric_only=True) contains :math:`24m(m-1)-8(m-1)` nodes, and
        pegasus_graph(m, nice_coordinates=True) contains :math:`24(m-1)^2` nodes.

    The maximum degree of these graph is 15, and counting formulas are more complicated
    for edges given most parameter settings.  Upper bounds are given below,

        pegasus_graph(1, fabric_only=False) has zero edges,
        pegasus_graph(m, fabric_only=False) has :math:`12*(15*(m-1)^2 + m - 3)` edges if m > 1

    Note that the above are valid with default offset parameters.

    A Pegasus lattice is a graph minor of a lattice similar to Chimera,
    where unit tiles are completely connected.  In the most generality,
    prelattice :math:`Q(N0,N1)` contains nodes of the form

        :math:`(i, j, 0, k)` with :math:`0 <= k < 2` [vertical nodes]

    and

        :math:`(i, j, 1, k)` with :math:`0 <= k < 2` [horizontal nodes]

    for :math:`0 <= i <= N0` and :math:`0 <= j < N1`; and edges of the form

        :math:`(i, j, u, k)` ~ :math:`(i+u, j+1-u, u, k)`  [external edges]

        :math:`(i, j, 0, k)` ~ :math:`(i, j, 1, k)` [internal edges]

        :math:`(i, j, u, 0)` ~ :math:`(i, j, u, 1)` [odd edges]

    The minor is specified by two lists of offsets; :math:`S0` and :math:`S1` of length
    :math:`L0` and :math:`L1` (where :math:`L0` and :math:`L1`, and the entries of
    :math:`S0` and :math:`S1`, must be divisible by 2).
    From these offsets, we construct our minor, a Pegasus lattice, by contracting
    the complete intervals of external edges::

        I(0, w, k, z) = [(L1*w + k, L0*z + S0[k] + r, 0, k % 2) for 0 <= r < L0]
        I(1, w, k, z) = [(L1*z + S1[k] + r, L0*w + k, 1, k % 2) for 0 <= r < L1]

    and deleting the prelattice nodes of any interval not fully contained in
    :math:`Q(N0, N1)`.

    This generator is specialized to :math:`L0 = L1 = 12`; :math:`N0 = N1 = 12m`.

    The notation :math:`(u, w, k, z)` is called the Pegasus index of a node in a Pegasus
    lattice.  The entries can be interpreted as following,

        :math:`u`: qubit orientation (0 = vertical, 1 = horizontal)

        :math:`w`: orthogonal major offset

        :math:`k`: orthogonal minor offset

        :math:`z`: parallel offset

    and the edges in the minor have the form

        :math:`(u, w, k, z)` ~ :math:`(u, w, k, z+1)` [external edges]

        :math:`(0, w0, k0, z0)` ~ :math:`(1, w1, k1, z1)` [internal edges, see below]

        :math:`(u, w, 2k, z)` ~ :math:`(u, w, 2k+1, z)` [odd edges]

    where internal edges only exist when::

        w1 = z0 + (1 if k1 < S0[k0] else 0), and
        z1 = w0 - (1 if k0 < S1[k1] else 0)

    linear indices are computed from pegasus indices by the formula::

        q = ((u * m + w) * 12 + k) * (m - 1) + z

    Parameters
    ----------
    m : int
        The size parameter for the Pegasus lattice.
    create_using : Graph, optional (default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable, optional (default None)
        Iterable of nodes in the graph. If None, calculated from m.
        Note that this list is used to remove nodes, so any nodes specified
        not in range(24 * m * (m-1)) will not be added.
    edge_list : iterable, optional (default None)
        Iterable of edges in the graph. If None, edges are generated as
        described above. The nodes in each edge must be integer-labeled in
        range(24 * m * (m-1)).
    data : bool, optional (default True)
        If True, each node has a pegasus_index attribute. The attribute
        is a 4-tuple Pegasus index as defined above. (if coordinates = True,
        we set a linear_index, which is an integer)
    coordinates : bool, optional (default False)
        If True, node labels are 4-tuple Pegasus indices.  Ignored if
        nice_coordinates is True
    offset_lists : pair of lists, optional (default None)
        Used to directly control the offsets, each list in the pair should
        have length 12, and contain even ints.  If offset_lists is not None,
        then offsets_index must be None.
    offsets_index : int, optional (default None)
        A number between 0 and 7 inclusive, to select a preconfigured
        set of topological parameters.  If both offsets_index and
        offset_lists are None, then we set offsets_index = 0.  At least
        one of these two parameters must be None.
    fabric_only: bool, optional (default True)
        The Pegasus graph, by definition, will have some disconnected
        components.  If this True, we will only construct nodes from the
        largest component.  Otherwise, the full disconnected graph will be
        constructed.  Ignored if edge_lists is not None or nice_coordinates
        is True
    nice_coordinates: bool, optional (default False)
        In the case that offsets_index = 0, generate the graph with a nicer
        coordinate system which is more compatible with Chimera addressing.
        These coordinates are 5-tuples taking the form (t, y, x, u, k) where
        0 <= x < M-1, 0 <= y < M-1, 0 <= u < 2, 0 <= k < 4 and 0 <= t < 3.
        For any given 0 <= t0 < 3, the subgraph of nodes with t = t0 has the
        structure of chimera(M-1, M-1, 4) with the addition of odd couplers.
        Supercedes both the fabric_only and coordinates parameters.
    """
    if offset_lists is None:
        offsets_descriptor = offsets_index = offsets_index or 0
        offset_lists = [
            [(2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6,), (6, 6, 6, 6, 2, 2, 2, 2, 10, 10, 10, 10,)],
            [(2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6,), (2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6,)],
            [(2, 2, 2, 2, 10, 10, 10, 10, 6, 6, 6, 6,), (10, 10, 10, 10, 6, 6, 6, 6, 2, 2, 2, 2,)],
            [(10, 10, 10, 10, 6, 6, 6, 6, 2, 2, 2, 2,), (10, 10, 10, 10, 6, 6, 6, 6, 2, 2, 2, 2,)],
            [(10, 10, 10, 10, 6, 6, 6, 6, 2, 2, 2, 2,), (2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10,)],
            [(6, 6, 2, 2, 2, 2, 10, 10, 10, 10, 6, 6,), (6, 6, 2, 2, 2, 2, 10, 10, 10, 10, 6, 6,)],
            [(6, 6, 2, 2, 2, 2, 10, 10, 10, 10, 6, 6,), (6, 6, 10, 10, 10, 10, 2, 2, 2, 2, 6, 6,)],
            [(6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,), (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,)],
        ][offsets_index]
    elif offsets_index is not None:
        raise DWaveNetworkXException("provide at most one of offsets_index and offset_lists")
    else:
        for ori in 0, 1:
            for x, y in zip(offset_lists[ori][::2], offset_lists[ori][1::2]):
                if x != y:
                    warnings.warn("The offets list you've provided is possibly non-physical.  Odd-coupled qubits should have the same value.")
        offsets_descriptor = offset_lists

    G = nx.empty_graph(0, create_using)

    G.name = "pegasus_graph(%s, %s)" % (m, offsets_descriptor)

    m1 = m - 1
    if nice_coordinates:
        if offsets_index != 0:
            raise NotImplementedError("nice coordinate system is only implemented for offsets_index 0")
        labels = 'nice'
        c2i = get_pegasus_to_nice_fn()
    elif coordinates:
        c2i = lambda *q: q
        labels = 'coordinate'
    else:
        labels = 'int'
        def c2i(u, w, k, z): return u * 12 * m * m1 + w * 12 * m1 + k * m1 + z

    construction = (("family", "pegasus"), ("rows", m), ("columns", m),
                    ("tile", 12), ("vertical_offsets", offset_lists[0]),
                    ("horizontal_offsets", offset_lists[1]), ("data", data),
                    ("labels", labels))

    G.graph.update(construction)

    max_size = m * (m - 1) * 24  # max number of nodes G can have

    if edge_list is None:
        if nice_coordinates:
            fabric_start = 4,8
            fabric_end = 8, 4
        elif fabric_only:
            fabric_start = min(s for s in offset_lists[1]), min(s for s in offset_lists[0])
            fabric_end = 12 - max(s for s in offset_lists[1]), 12 - max(s for s in offset_lists[0])
        else:
            fabric_end = fabric_start = 0, 0

        G.add_edges_from((c2i(u, w, k, z), c2i(u, w, k, z + 1))
                         for u in (0, 1)
                         for w in range(m)
                         for k in range(fabric_start[u] if w == 0 else 0, 12 - (fabric_end[u] if w == m1 else 0))
                         for z in range(m1 - 1))

        G.add_edges_from((c2i(u, w, k, z), c2i(u, w, k + 1, z))
                         for u in (0, 1)
                         for w in range(m)
                         for k in range(fabric_start[u] if w == 0 else 0, 12 - (fabric_end[u] if w == m1 else 0), 2)
                         for z in range(m1))

        off0, off1 = offset_lists
        def qfilter(u, w, k, z):
            if w == 0: return k >= fabric_start[u]
            if w == m1: return k < 12-fabric_end[u]
            return True
        def efilter(e): return qfilter(*e[0]) and qfilter(*e[1])

        internal_couplers = (((0, w, k, z), (1, z + (kk < off0[k]), kk, w - (k < off1[kk])))
                         for w in range(m)
                         for kk in range(12)
                         for k in range(0 if w else off1[kk], 12 if w < m1 else off1[kk])
                         for z in range(m1))
        G.add_edges_from((c2i(*e[0]), c2i(*e[1])) for e in internal_couplers if efilter(e))

    else:
        G.add_edges_from(edge_list)

    if node_list is not None:
        nodes = set(node_list)
        G.remove_nodes_from(set(G) - nodes)
        G.add_nodes_from(nodes)  # for singleton nodes

    if data:
        v = 0
        for u in range(2):
            for w in range(m):
                for k in range(12):
                    for z in range(m1):
                        q = u, w, k, z
                        if nice_coordinates:
                            p = c2i(*q)
                            if p in G:
                                G.nodes[p]['linear_index'] = v
                                G.nodes[p]['pegasus_index'] = q
                        elif coordinates:
                            if q in G:
                                G.nodes[q]['linear_index'] = v
                        else:
                            if v in G:
                                G.nodes[v]['pegasus_index'] = q
                        v += 1

    return G


def get_tuple_fragmentation_fn(pegasus_graph):
    """
    Returns a fragmentation function that is specific to pegasus_graph. This fragmentation function,
    fragment_tuple(..), takes in a list of Pegasus qubit coordinates and returns their corresponding
    K2,2 Chimera fragment coordinates.

    Details on the returned function, fragment_tuple(list_of_pegasus_coordinates):
        Each Pegasus qubit is split into six fragments. If edges are drawn between adjacent
        fragments and drawn between fragments that are connected by an existing Pegasus coupler, we
        can see that a K2,2 Chimera graph is formed.

        The K2,2 Chimera graph uses a coordinate system with an origin at the upper left corner of
        the graph.
            y: number of vertical fragments from the top-most row
            x: number of horizontal fragments from the left-most column
            u: 1 if it belongs to a horizontal qubit, 0 otherwise
            r: fragment index on the K2,2 shore

    Parameters
    ----------
    pegasus_graph: networkx.graph
        A pegasus graph

    Returns
    -------
    fragment_tuple(pegasus_coordinates): a function
        A function that accepts a list of pegasus coordinates and returns a list of their
        corresponding K2,2 Chimera coordinates.
    """
    horizontal_offsets = pegasus_graph.graph['horizontal_offsets']
    vertical_offsets = pegasus_graph.graph['vertical_offsets']

    # Note: we are returning a fragmentation function rather than fragmenting the pegasus
    # coordinates ourselves because:
    #   (1) We don't want the user to have to deal with Pegasus horizontal/vertical offsets directly.
    #       (i.e. Don't want fragment_tuple(pegasus_coord, vertical_offset, horizontal_offset))
    #   (2) We don't want the user to have to pass entire Pegasus graph each time they want to
    #       fragment some pegasus coordinates.
    #       (i.e. Don't want fragment_tuple(pegasus_coord, pegasus_graph))
    def fragment_tuple(pegasus_coords):
        fragments = []
        for u, w, k, z in pegasus_coords:
            # Determine offset
            offset = horizontal_offsets if u else vertical_offsets
            offset = offset[k]

            # Find the base (i.e. zeroth) Chimera fragment of this pegasus coordinate
            x0 = (z * 12 + offset) // 2
            y = (w * 12 + k) // 2
            r = k % 2
            base = [0, 0, u, r]

            # Generate the six fragments associated with this pegasus coordinate
            for x in range(x0, x0 + 6):
                base[u] = x
                base[1 - u] = y
                fragments.append(tuple(base))

        return fragments

    return fragment_tuple


def get_tuple_defragmentation_fn(pegasus_graph):
    """
    Returns a de-fragmentation function that is specific to pegasus_graph. The returned
    de-fragmentation function, defragment_tuple(..), takes in a list of K2,2 Chimera coordinates and
    returns the corresponding list of unique pegasus coordinates.

    Details on the returned function, defragment_tuple(list_of_chimera_fragment_coordinates):
        Each Pegasus qubit is split into six fragments. If edges are drawn between adjacent
        fragments and drawn between fragments that are connected by an existing Pegasus coupler, we
        can see that a K2,2 Chimera graph is formed.

        The K2,2 Chimera graph uses a coordinate system with an origin at the upper left corner of
        the graph.
            y: number of vertical fragments from the top-most row
            x: number of horizontal fragments from the left-most column
            u: 1 if it belongs to a horizontal qubit, 0 otherwise
            r: fragment index on the K2,2 shore

        The defragment_tuple(..) takes in the list of Chimera fragments and returns a list of their
        corresponding Pegasus qubit coordinates. Note that the returned list has a unique set of
        Pegasus coordinates.

    Parameters
    ----------
    pegasus_graph: networkx.graph
        A Pegasus graph

    Returns
    -------
    defragment_tuple(chimera_coordinates): a function
        A function that accepts a list of chimera coordinates and returns a set of their
        corresponding Pegasus coordinates.
    """
    horizontal_offsets = pegasus_graph.graph['horizontal_offsets']
    vertical_offsets = pegasus_graph.graph['vertical_offsets']

    # Note: we are returning a defragmentation function rather than defragmenting the chimera
    # fragments ourselves because:
    #   (1) We don't want the user to have to deal with Pegasus horizontal/vertical offsets directly.
    #       (i.e. Don't want defragment_tuple(chimera_coord, vertical_offset, horizontal_offset))
    #   (2) We don't want the user to have to pass entire Pegasus graph each time they want to
    #       defragment some chimera coordinates.
    #       (i.e. Don't want defragment_tuple(chimera_coord, pegasus_graph))
    def defragment_tuple(chimera_coords):
        pegasus_coords = []
        for y, x, u, r in chimera_coords:
            # Set up shifts and offsets
            shifts = [x, y]
            offsets = horizontal_offsets if u else vertical_offsets

            # Determine number of tiles and track number
            w, k = divmod(2 * shifts[u] + r, 12)

            # Determine qubit index on track
            x0 = shifts[1-u] * 2 - offsets[k]
            z = x0 // 12

            pegasus_coords.append((u, w, k, z))

        # Several chimera coordinates may map to the same pegasus coordinate, hence, apply set(..)
        return list(set(pegasus_coords))

    return defragment_tuple


# Developer note: we could implement a function that creates the iter_*_to_* and
# iter_*_to_*_pairs methods just-in-time, but there are a small enough number
# that for now it makes sense to do them by hand.
class pegasus_coordinates(object):
    def __init__(self, m):
        """Provides coordinate converters for the Pegasus indexing schemes.

        Parameters
        ----------
        m : int
            The size parameter for the Pegasus lattice.

        See also
        --------
        :func:`.pegasus_graph` for a description of the various coordinate
        conventions.

        """

        self.args = m, m - 1

    def pegasus_to_linear(self, q):
        """Convert a 4-term Pegasus coordinate into a linear index."""
        u, w, k, z = q
        m, m1 = self.args
        return ((m * u + w) * 12 + k) * m1 + z

    def linear_to_pegasus(self, r):
        """Convert a linear index into a 4-term Pegasus coordinate."""
        m, m1 = self.args
        r, z = divmod(r, m1)
        r, k = divmod(r, 12)
        u, w = divmod(r, m)
        return u, w, k, z

    @staticmethod
    def nice_to_pegasus(n):
        """Convert a 5-term nice coordinate into a 4-term Pegasus coordinate.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        t, y, x, u, k = n

        if t == 0:
            return (u, y+1 if u else x, 4+k if u else 4+k, x if u else y)
        elif t == 1:
            return (u, y+1 if u else x, k if u else 8+k, x if u else y)
        elif t == 2:
            return (u, y if u else x + 1, 8+k if u else k, x if u else y)

        # can happen when t is a float for instance
        raise ValueError("invalid Nice coordinate: {}".format(n))

    @staticmethod
    def pegasus_to_nice(p):
        """Convert a 4-term Pegasus coordinate to a 5-term nice coordinate.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        u, w, k, z = p

        t = (2-u-(2*u-1)*(k//4)) % 3

        if t == 0:
            return (0, w-1 if u else z, z if u else w, u, k-4 if u else k-4)
        elif t == 1:
            return (1, w-1 if u else z, z if u else w, u, k if u else k-8)
        elif t == 2:
            return (2, w if u else z, z if u else w-1, u, k-8 if u else k)

        # can happen when given floats for instance
        raise ValueError('invalid Pegasus coordinates')

    def linear_to_nice(self, r):
        """Convert a linear index into a 5-term nice coordinate."""
        return self.pegasus_to_nice(self.linear_to_pegasus(r))

    def nice_to_linear(self, n):
        """Convert a 5-term nice coordinate into a linear index."""
        return self.pegasus_to_linear(self.nice_to_pegasus(n))

    def iter_pegasus_to_linear(self, qlist):
        """Return an iterator converting a sequence of 4-term Pegasus
        coordinates to linear indices.
        """
        m, m1 = self.args
        for (u, w, k, z) in qlist:
            yield ((m * u + w) * 12 + k) * m1 + z

    def iter_linear_to_pegasus(self, rlist):
        """Return an iterator converting a sequence of linear indices to 4-term
        Pegasus coordinates.
        """
        m, m1 = self.args
        for r in rlist:
            r, z = divmod(r, m1)
            r, k = divmod(r, 12)
            u, w = divmod(r, m)
            yield u, w, k, z

    @classmethod
    def iter_nice_to_pegasus(cls, nlist):
        """Return an iterator converting a sequence of 5-term nice coordinates
        to 4-term Pegasus coordinates.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        for n in nlist:
            yield cls.nice_to_pegasus(n)

    @classmethod
    def iter_pegasus_to_nice(cls, plist):
        """Return an iterator converting a sequence of 4-term Pegasus
        coordinates to 5-term nice coordinates.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        for p in plist:
            yield cls.pegasus_to_nice(p)

    def iter_linear_to_nice(self, rlist):
        """Return an iterator converting a sequence of linear indices to 5-term
        nice coordinates.
        """
        for r in rlist:
            yield self.linear_to_nice(r)

    def iter_nice_to_linear(self, nlist):
        """Return an iterator converting a sequence of 5-term nice coordinates
        to linear indices.
        """
        for n in nlist:
            yield self.nice_to_linear(n)

    @staticmethod
    def _pair_repack(f, plist):
        """Flattens a sequence of pairs to pass through `f`, and then
        re-pairs the result.
        """
        ulist = f(u for p in plist for u in p)
        for u in ulist:
            v = next(ulist)
            yield u, v

    def iter_pegasus_to_linear_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of 4-term Pegasus
        coordinates to pairs of linear indices.
        """
        return self._pair_repack(self.iter_pegasus_to_linear, plist)

    def iter_linear_to_pegasus_pairs(self, plist):
        """Return an iterator converting a sequence of pairs of linear indices
        to pairs of 4-term Pegasus coordinates.
        """
        return self._pair_repack(self.iter_linear_to_pegasus, plist)

    @classmethod
    def iter_nice_to_pegasus_pairs(cls, nlist):
        """Return an iterator converting a sequence of pairs of 5-term nice
        coordinates to pairs of 4-term Pegasus coordinates.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        return cls._pair_repack(cls.iter_nice_to_pegasus, nlist)

    @classmethod
    def iter_pegasus_to_nice_pairs(cls, plist):
        """Return an iterator converting a sequence of pairs of 4-term Pegasus
        coordinates to pairs of 5-term nice coordinates.

        Note that this method does not depend on the size of the Pegasus
        lattice.
        """
        return cls._pair_repack(cls.iter_pegasus_to_nice, plist)

    def iter_linear_to_nice_pairs(self, rlist):
        """Return an iterator converting a sequence of pairs of linear indices
        to pairs of 5-term nice coordinates.
        """
        return self._pair_repack(self.iter_linear_to_nice, rlist)

    def iter_nice_to_linear_pairs(self, nlist):
        """Return an iterator converting a sequence of pairs of 5-term nice
        coordinates to pairs of linear indices.
        """
        return self._pair_repack(self.iter_nice_to_linear, nlist)

    def int(self, q):
        """Deprecated alias of `pegasus_to_linear`."""
        msg = ('pegasus_coordinates.int is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.pegasus_to_linear instead')
        warnings.warn(msg, DeprecationWarning)
        return self.pegasus_to_linear(q)

    def tuple(self, r):
        """Deprecated alias for `linear_to_pegasus`."""
        msg = ('pegasus_coordinates.tuple is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.linear_to_pegasus instead')
        warnings.warn(msg, DeprecationWarning)
        return self.linear_to_pegasus(r)

    def ints(self, qlist):
        """Deprecated alias for `iter_pegasus_to_linear`."""
        msg = ('pegasus_coordinates.ints is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.iter_pegasus_to_linear instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_pegasus_to_linear(qlist)

    def tuples(self, rlist):
        """Deprecated alias for `iter_linear_to_pegasus`."""
        msg = ('pegasus_coordinates.tuples is deprecated and will be removed in '
               'dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.iter_linear_to_pegasus instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_linear_to_pegasus(rlist)

    def int_pairs(self, plist):
        """Deprecated alias for `iter_pegasus_to_linear_pairs`."""
        msg = ('pegasus_coordinates.int_pairs is deprecated and will be removed'
               ' in dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.iter_pegasus_to_linear_pairs instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_pegasus_to_linear_pairs(plist)

    def tuple_pairs(self, plist):
        """Deprecated alias for `iter_linear_to_pegasus_pairs`."""
        msg = ('pegasus_coordinates.tuple_pairs is deprecated and will be removed'
               ' in dwave-networkx 0.9.0, please use '
               'pegasus_coordinates.iter_linear_to_pegasus_pairs instead')
        warnings.warn(msg, DeprecationWarning)
        return self.iter_linear_to_pegasus_pairs(plist)


# maintained for backwards compatibility
def get_pegasus_to_nice_fn(*args, **kwargs):
    """
    Returns a coordinate translation function from the 4-term pegasus_index
    coordinates to the 5-term "nice" coordinates.

    Details on the returned function, pegasus_to_nice(u,w,k,z)
        Inputs are 4-tuples of ints, return is a 5-tuple of ints.  See
        pegasus_graph for description of the pegasus_index and "nice"
        coordinate systems.

    Returns
    -------
    pegasus_to_nice_fn(chimera_coordinates): a function
        A function that accepts augmented chimera coordinates and returns corresponding 
        Pegasus coordinates.
    """
    if args or kwargs:
        msg = "get_pegasus_to_nice_fn does not need / use parameters anymore"
        warnings.warn(msg, DeprecationWarning)

    msg = ('get_pegasus_to_nice_fn is deprecated and will be removed in '
           'dwave-networkx 0.9.0, please use '
           'pegasus_coordinates.pegasus_to_nice instead')
    warnings.warn(msg, DeprecationWarning)

    return lambda *args: pegasus_coordinates.pegasus_to_nice(args)


# maintained for backwards compatibility
def get_nice_to_pegasus_fn(*args, **kwargs):
    """
    Returns a coordinate translation function from the 5-term "nice"
    coordinates to the 4-term pegasus_index coordinates.

    Details on the returned function, nice_to_pegasus(t, y, x, u, k)
        Inputs are 5-tuples of ints, return is a 4-tuple of ints.  See
        pegasus_graph for description of the pegasus_index and "nice"
        coordinate systems.

    Returns
    -------
    nice_to_pegasus_fn(pegasus_coordinates): a function
        A function that accepts Pegasus coordinates and returns the corresponding 
        augmented chimera coordinates
    """
    if args or kwargs:
        msg = "get_pegasus_to_nice_fn does not need / use parameters anymore"
        warnings.warn(msg, DeprecationWarning)

    msg = ('get_nice_to_pegasus_fn is deprecated and will be removed in '
           'dwave-networkx 0.9.0, please use '
           'pegasus_coordinates.nice_to_pegasus instead')
    warnings.warn(msg, DeprecationWarning)

    return lambda *args: pegasus_coordinates.nice_to_pegasus(args)
