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
Generators for some graphs derived from the D-Wave System.
"""
import re

import networkx as nx

from dwave_networkx.exceptions import DWaveNetworkXException
import warnings

from itertools import product
from .chimera import _chimera_coordinates_cache
from .common import _add_compatible_edges, _add_compatible_nodes, _add_compatible_terms

__all__ = ['pegasus_graph',
           'pegasus_coordinates',
           'pegasus_sublattice_mappings',
           'pegasus_torus',
           ]

def pegasus_graph(m, create_using=None, node_list=None, edge_list=None, data=True,
                  offset_lists=None, offsets_index=None, coordinates=False, fabric_only=True,
                  nice_coordinates=False, check_node_list=False, check_edge_list=False):
    """
    Creates a Pegasus graph with size parameter `m`.

    Parameters
    ----------
    m : int
        Size parameter for the Pegasus lattice.
    create_using : Graph, optional (default None)
        If provided, this graph is cleared of nodes and edges and filled
        with the new graph. Usually used to set the type of the graph.
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph.  The nodes should typically be 
        compatible with the requested lattice shape parameters and coordinate 
        system, incompatible nodes are accepted unless you set :code:`check_node_list=True`. 
        If not specified, calculated from ``m``, ``fabric_only``, 
        ``nice_coordinates``, ``offset_lists`` and ``offset_index`` and
        ``coordinates`` per the topology description below.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. Edges must be 2-tuples of the nodes 
        specified in ``node_list``, or calculated from ``m``, ``fabric_only``, 
        ``nice_coordinates``, ``offset_lists`` and ``offset_index`` and
        ``coordinates`` per the topology description below; incompatible edges 
        are ignored unless you set :code:`check_edge_list=True`. If not 
        specified, all edges compatible with the ``node_list`` and topology 
        description are included.
    data : bool, optional (default :code:`True`)
        If :code:`True`, each node has a pegasus_index attribute. The attribute
        is a 4-tuple Pegasus index as defined below. If the `coordinates` 
        parameter is :code:`True`, a linear_index, which is an integer, is used.
    coordinates : bool, optional (default :code:`False`)
        If :code:`True`, node labels are 4-tuple Pegasus indices. Ignored if the
        `nice_coordinates` parameter is :code:`True`.
    offset_lists : pair of lists, optional (default None)
        Directly controls the offsets. Each list in the pair must have length 12
        and contain even ints.  If `offset_lists` is not None, the `offsets_index`
        parameter must be None.
    offsets_index : int, optional (default None)
        A number between 0 and 7, inclusive, that selects a preconfigured
        set of topological parameters. If both the `offsets_index` and
        `offset_lists` parameters are None, the `offsets_index` parameters is set
        to zero. At least one of these two parameters must be None.
    fabric_only: bool, optional (default :code:`True`)
        The Pegasus graph, by definition, has some disconnected
        components.  If :code:`True`, the generator only constructs nodes from the
        largest component. If :code:`False`, the full disconnected graph is
        constructed. Ignored if the `edge_lists` parameter is not None or
        `nice_coordinates` is :code:`True`
    nice_coordinates: bool, optional (default :code:`False`)
        If the `offsets_index` parameter is 0, the graph uses a "nicer"
        coordinate system, more compatible with Chimera addressing.
        These coordinates are 5-tuples taking the form :math:`(t, y, x, u, k)` where
        :math:`0 <= x < M-1`, :math:`0 <= y < M-1`, :math:`0 <= u < 2`,
        :math:`0 <= k < 4`, and :math:`0 <= t < 3`.
        For any given :math:`0 <= t0 < 3`, the subgraph of nodes with :math:`t = t0`
        has the structure of `chimera(M-1, M-1, 4)` with the addition of odd couplers.
        Supercedes both the `fabric_only` and `coordinates` parameters.
    check_node_list : bool (optional, default :code:`False`)
        If :code:`True`, the ``node_list`` elements are checked for compatibility with
        the graph topology and node labeling conventions, an error is thrown
        if any node is incompatible or duplicates exist.
        In other words, only node lists that specify subgraphs of the default 
        (full yield) graph are permitted. An exception is allowed if 
        ``check_edge_list=False``, in which case any node in ``edge_list`` is treated as valid.
    check_edge_list : bool (optional, default :code:`False`)
        If :code:`True`, the edge_list elements are checked for compatibility with
        the graph topology and node labeling conventions, an error is thrown
        if any edge is incompatible or duplicates exist. 
        In other words, only edge_lists that specify subgraphs of the default 
        (full yield) graph are permitted.

    Returns
    -------
    G : NetworkX Graph
        A Pegasus lattice for size parameter `m`.


    The maximum degree of this graph is 15. The number of nodes depends on multiple
    parameters; for example,

        * :code:`pegasus_graph(1)`: zero nodes
        * :code:`pegasus_graph(m, fabric_only=False)`: :math:`24m(m-1)` nodes
        * :code:`pegasus_graph(m, fabric_only=True)`: :math:`24m(m-1)-8(m-1)` nodes
        * :code:`pegasus_graph(m, nice_coordinates=True)`: :math:`24(m-1)^2` nodes

    Counting formulas for edges have a complicated dependency on parameter settings.
    Some example upper bounds are:

        * :code:`pegasus_graph(1, fabric_only=False)`: zero edges
        * :code:`pegasus_graph(m, fabric_only=False)`: :math:`12*(15*(m-1)^2 + m - 3)`
          edges if :math:`m > 1`

    Note that the formulas above are valid for default offset parameters.

    A Pegasus lattice is a graph minor of a lattice similar
    to Chimera, where unit tiles are completely connected. In its most general
    definition, prelattice :math:`Q(N0,N1)` contains nodes of the form

        * vertical nodes: :math:`(i, j, 0, k)` with :math:`0 <= k < 2`
        * horizontal nodes: :math:`(i, j, 1, k)` with :math:`0 <= k < 2`

    for :math:`0 <= i <= N0` and :math:`0 <= j < N1`, and edges of the form

        * external: :math:`(i, j, u, k)` ~ :math:`(i+u, j+1-u, u, k)`
        * internal: :math:`(i, j, 0, k)` ~ :math:`(i, j, 1, h)`
        * odd: :math:`(i, j, u, 0)` ~ :math:`(i, j, u, 1)`

    Given two lists of offsets, :math:`S0` and :math:`S1`, of length
    :math:`L0` and :math:`L1`, where both lengths and values must be divisible by
    2, the minor---a Pegasus lattice---is constructed by contracting the complete
    intervals of external edges::

        I(0, w, k, z) = [(L1*w + k, L0*z + S0[k] + r, 0, k % 2) for 0 <= r < L0]
        I(1, w, k, z) = [(L1*z + S1[k] + r, L0*w + k, 1, k % 2) for 0 <= r < L1]

    and deleting the prelattice nodes of any interval not fully contained in
    :math:`Q(N0, N1)`.

    This generator, 'pegasus_graph()', is specialized for the minor constructed by
    prelattice and offset parameters :math:`L0 = L1 = 12` and :math:`N0 = N1 = 12m`.

    The *Pegasus index* of a node in a Pegasus lattice, :math:`(u, w, k, z)`, can be
    interpreted as:

        * :math:`u`: qubit orientation (0 = vertical, 1 = horizontal)
        * :math:`w`: orthogonal major offset
        * :math:`k`: orthogonal minor offset
        * :math:`z`: parallel offset

    Edges in the minor have the form

        * external: :math:`(u, w, k, z)` ~ :math:`(u, w, k, z+1)`
        * internal: :math:`(0, w0, k0, z0)` ~ :math:`(1, w1, k1, z1)`
        * odd: :math:`(u, w, 2k, z)` ~ :math:`(u, w, 2k+1, z)`

    where internal edges only exist when

        1. w1 = z0 + (1 if k1 < S0[k0] else 0)
        2. z1 = w0 - (1 if k0 < S1[k1] else 0)

    Linear indices are computed from Pegasus indices by the formula::

        q = ((u * m + w) * 12 + k) * (m - 1) + z


    Examples
    ========
    >>> G = dnx.pegasus_graph(2, nice_coordinates=True)
    >>> G.nodes(data=True)[(0, 0, 0, 0, 0)]    # doctest: +SKIP
    {'linear_index': 4, 'pegasus_index': (0, 0, 4, 0)}


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
        pegasus_to_nice = pegasus_coordinates.pegasus_to_nice
        nice_to_pegasus = pegasus_coordinates.nice_to_pegasus
        label = lambda *q: pegasus_to_nice(q)
    elif coordinates:
        label = lambda *q: q
        labels = 'coordinate'
    else:
        labels = 'int'
        def label(u, w, k, z):
            return u * 12 * m * m1 + w * 12 * m1 + k * m1 + z

    construction = (("family", "pegasus"), ("rows", m), ("columns", m),
                    ("tile", 12), ("vertical_offsets", offset_lists[0]),
                    ("horizontal_offsets", offset_lists[1]), ("data", data),
                    ("labels", labels))

    G.graph.update(construction)

    max_size = m * (m - 1) * 24  # max number of nodes G can have

    if edge_list is None:
        check_edge_list = False
    if node_list is None:
        check_node_list = False
    
    if edge_list is None or check_edge_list is True:
        if nice_coordinates:
            fabric_start = 4,8
            fabric_end = 8, 4
        elif fabric_only:
            fabric_start = min(s for s in offset_lists[1]), min(s for s in offset_lists[0])
            fabric_end = 12 - max(s for s in offset_lists[1]), 12 - max(s for s in offset_lists[0])
        else:
            fabric_end = fabric_start = 0, 0

        G.add_edges_from((label(u, w, k, z), label(u, w, k, z + 1))
                         for u in (0, 1)
                         for w in range(m)
                         for k in range(fabric_start[u] if w == 0 else 0, 12 - (fabric_end[u] if w == m1 else 0))
                         for z in range(m1 - 1))

        G.add_edges_from((label(u, w, k, z), label(u, w, k + 1, z))
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
        G.add_edges_from((label(*e[0]), label(*e[1])) for e in internal_couplers if efilter(e))

        if edge_list is not None:
            _add_compatible_edges(G, edge_list)
    else:
        if check_node_list or node_list is None:
            G.add_nodes_from(label(u, w, k, z) for u in range(2)
                                  for w in range(m)
                                  for k in range(12)
                                  for z in range(m-1))
        G.add_edges_from(edge_list)

    if node_list is not None:
        if check_node_list:
            _add_compatible_nodes(G, node_list)
        else:
            nodes = set(node_list)
            G.remove_nodes_from(set(G) - nodes)
            G.add_nodes_from(nodes)  # for singleton nodes

    if data:
        v = 0
        if nice_coordinates:
            def fill_data():
                q = (u, w, k, z)
                d = get_node_data(pegasus_to_nice(q))
                if d is not None:
                    d['linear_index'] = v
                    d['pegasus_index'] = q
        elif coordinates:
            def fill_data():
                d = get_node_data((u, w, k, z))
                if d is not None:
                    d['linear_index'] = v
        else:
            def fill_data():
                d = get_node_data(v)
                if d is not None:
                    d['pegasus_index'] = (u, w, k, z)

        get_node_data = G.nodes.get
        for u in range(2):
            for w in range(m):
                for k in range(12):
                    for z in range(m1):
                        fill_data()
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
            fz0 = (z*12 + offset) // 2 # first fragment's z-coordinate
            fw = (w*12 + k) // 2 # fragment w-coordinate
            fk = k&1 # fragment k-index
            base = [fw, 0, u, fk] if u else [0, fw, u, fk]

            # Generate the six fragments associated with this pegasus coordinate
            for fz in range(fz0, fz0 + 6):
                base[u] = fz
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
            z = (shifts[1-u] * 2 - offsets[k]) // 12

            pegasus_coords.append((u, w, k, z))

        # Several chimera coordinates may map to the same pegasus coordinate, hence, apply set(..)
        return list(set(pegasus_coords))

    return defragment_tuple


def fragmented_edges(pegasus_graph):
    """
    Generator for the edges contained in a Chimera graph obtained by splitting each Pegasus node into
    six Chimera nodes.  If the Pegasus graph has size parameter m, then the derived graph will be a
    subgraph of Chimera(6m, 6m, 2) -- that is, a Chimera graph with K2,2 unit tiles.

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
    (coord0, coord1), ... : an iterator of tuples
        Yields the edges contained in the Chimera graph derived from the "fragmentation" construction
    """
    offsetlist = [pegasus_graph.graph['vertical_offsets'], pegasus_graph.graph['horizontal_offsets']]
    offsets = {(u, k): offsetlist[u][k] for u in (0, 1) for k in range(12)}

    if pegasus_graph.graph['labels'] == 'nice':
        coords = pegasus_coordinates.nice_to_pegasus
    elif pegasus_graph.graph['labels'] == 'int':
        coords = pegasus_coordinates(pegasus_graph.graph['rows']).linear_to_pegasus
    else:
        coords = lambda z: z

    # first, we generate the edges internal to the fragments corresponding to a node
    for q in pegasus_graph.nodes():
        u, w, k, z = coords(q)
        # copied from get_tuple_fragmentation_fn and slightly optimized
        offset = offsets[u, k]
        fz0 = (z*12 + offset) // 2 # first fragment z-coordinate
        fw = (w*12 + k) // 2 # fragment w-coordinate
        base = [fw, fz0, u, k&1] if u else [fz0, fw, u, k&1]
        prev = tuple(base)
        for fz in range(fz0+1, fz0+6):
            base[u] = fz
            curr = tuple(base)
            yield (prev, curr)
            prev = curr

    # now for the thinky part: for each Pegasus edge, generate the corresponding Chimera edge
    # we skip the "odd-coupler" edges because they don't exist in Chimera
    for q0, q1 in pegasus_graph.edges():
        u0, w0, k0, z0 = coords(q0)
        u1, w1, k1, z1 = coords(q1)
        if u0 == u1:
            if k0 == k1:
                # this is an external edge -- we could probably do some hijinks to fold this in
                # with the nodes loop, but cost/benefit doesn't support it right now
                offset = offsets[u0, k0]
                fz = (min(z0, z1)*12 + offset) // 2 # first fragment z-coordinate in the pair
                fw = (w0*12 + k0) // 2 # fragment w-coordinate for both qubits
                fk = k0&1 # fragment k-index
                if u0:
                    yield ((fw, fz+5, u0, fk), (fw, fz+6, u0, fk))
                else:
                    yield ((fz+5, fw, u0, fk), (fz+6, fw, u0, fk))

            # else: this is an odd edge; yield nothing
        else:
            # this may look a little magical -- we're looking for an edge of the form
            # (fy, fx, u0, fk0), (fy, fx, u1, fk1)
            # where (fy, fx) are the first two coordinates of both the fragments of (u0, w0, k0,z0),
            # (fy, fx) in [(fz0+0, fw0), (fz0+1, fw0), ..., (fz0+5, fw0)]
            # and also the the fragments of (u1, w1, k1, z1):
            # (fy, fx) in [(fw1, fz1+0), (fw1, fz1+1), ..., (fw1, fz1+5)]
            # (see get_tuple_fragmentation_fn to see the fragment generator)
            # with the assumption that an intersection exists: it can only be located at (fw0, fw1)
            # since those coordinates are constant in the respective intervals.  Thus, we get to
            # skip looking up the offsets.  Magic?  No, Math!
            fw0 = (w0*12 + k0) // 2
            fw1 = (w1*12 + k1) // 2
            if u0:
                yield ((fw0, fw1, u0, k0&1), (fw0, fw1, u1, k1&1))
            else:
                yield ((fw1, fw0, u0, k0&1), (fw1, fw0, u1, k1&1))


# Developer note: we could implement a function that creates the iter_*_to_* and
# iter_*_to_*_pairs methods just-in-time, but there are a small enough number
# that for now it makes sense to do them by hand.
class pegasus_coordinates(object):
    """Provides coordinate converters for the Pegasus indexing schemes.

    Parameters
    ----------
    m : int
        Size parameter for the Pegasus lattice.

    See also
    --------
    :func:`.pegasus_graph` : Describes the various coordinate conventions.

    """
    def __init__(self, m):

        self.args = m, m - 1

    def pegasus_to_linear(self, q):
        """Convert a 4-term Pegasus coordinate into a linear index.

        Parameters
        ----------
        q : 4-tuple
            Pegasus indices.

        Examples
        --------
        >>> dnx.pegasus_coordinates(2).pegasus_to_linear((0, 0, 4, 0))
        4
        """
        u, w, k, z = q
        m, m1 = self.args
        return ((m * u + w) * 12 + k) * m1 + z

    def linear_to_pegasus(self, r):
        """Convert a linear index into a 4-term Pegasus coordinate.

        Parameters
        ----------
        r : int
            Linear index.

        Examples
        --------
        >>> dnx.pegasus_coordinates(2).linear_to_pegasus(4)
        (0, 0, 4, 0)

        """
        m, m1 = self.args
        r, z = divmod(r, m1)
        r, k = divmod(r, 12)
        u, w = divmod(r, m)
        return u, w, k, z

    @staticmethod
    def nice_to_pegasus(n):
        """Convert a 5-term nice coordinate into a 4-term Pegasus coordinate.

        Parameters
        ----------
        n : 5-tuple
            Nice coordinate.

        Examples
        --------
        >>> dnx.pegasus_coordinates.nice_to_pegasus((0, 0, 0, 0, 0))
        (0, 0, 4, 0)

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

        Parameters
        ----------
        p : 4-tuple
            Pegasus coordinate.

        Examples
        --------
        >>> dnx.pegasus_coordinates.pegasus_to_nice((0, 0, 4, 0))
        (0, 0, 0, 0, 0)

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
        """Convert a linear index into a 5-term nice coordinate.

        Parameters
        ----------
        r : int
            Linear index.

        Examples
        --------
        >>> dnx.pegasus_coordinates(2).linear_to_nice(4)
        (0, 0, 0, 0, 0)
        """
        return self.pegasus_to_nice(self.linear_to_pegasus(r))

    def nice_to_linear(self, n):
        """Convert a 5-term nice coordinate into a linear index.

        Parameters
        ----------
        n : 5-tuple
            Nice coordinate.

        Examples
        --------
        >>> dnx.pegasus_coordinates(2).nice_to_linear((0, 0, 0, 0, 0))
        4
        """
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

    def graph_to_linear(self, g):
        """Return a copy of the graph g relabeled to have linear indices"""
        labels = g.graph.get('labels')
        if labels == 'int':
            return g.copy()
        elif labels == 'coordinate':
            nodes = self.iter_pegasus_to_linear(g)
            edges = self.iter_pegasus_to_linear_pairs(g.edges)
        elif labels == 'nice':
            nodes = self.iter_nice_to_linear(g)
            edges = self.iter_nice_to_linear_pairs(g.edges)
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.pegasus_graph."
            )

        return pegasus_graph(
            g.graph['rows'],
            node_list=nodes,
            edge_list=edges,
            data=g.graph['data'],
            offset_lists=(
                g.graph["vertical_offsets"],
                g.graph["horizontal_offsets"],
            ),
        )

    def graph_to_pegasus(self, g):
        """Return a copy of the graph g relabeled to have pegasus coordinates"""
        labels = g.graph.get('labels')
        if labels == 'int':
            nodes = self.iter_linear_to_pegasus(g)
            edges = self.iter_linear_to_pegasus_pairs(g.edges)
        elif labels == 'coordinate':
            return g.copy()
        elif labels == 'nice':
            nodes = self.iter_nice_to_pegasus(g)
            edges = self.iter_nice_to_pegasus_pairs(g.edges)
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.pegasus_graph."
            )

        return pegasus_graph(
            g.graph['rows'],
            node_list=nodes,
            edge_list=edges,
            data=g.graph['data'],
            coordinates=True,
            offset_lists=(
                g.graph["vertical_offsets"],
                g.graph["horizontal_offsets"],
            ),
        )

    def graph_to_nice(self, g):
        """Return a copy of the graph p relabeled to have nice coordinates"""
        labels = g.graph.get('labels')
        if labels == 'int':
            nodes = self.iter_linear_to_nice(g)
            edges = self.iter_linear_to_nice_pairs(g.edges)
        elif labels == 'coordinate':
            nodes = self.iter_pegasus_to_nice(g)
            edges = self.iter_pegasus_to_nice_pairs(g.edges)
        elif labels == 'nice':
            return g.copy()
        else:
            raise ValueError(
                f"Node labeling {labels} not recognized.  Input must be generated by dwave_networkx.pegasus_graph."
            )

        return pegasus_graph(
            g.graph['rows'],
            node_list=nodes,
            edge_list=edges,
            data=g.graph['data'],
            nice_coordinates=True,
            offsets_index = 0,
        )

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


def _chimera_pegasus_sublattice_mapping(source_to_chimera, nice_to_target, offset):
    """Constructs a mapping from a Chimera graph to a Pegasus graph, via an offset.
    This function is used by pegasus_sublattice_mappings, and serves to 
    construct a closure that is stable under iteration therein.

    Parameters
    ----------
        source_to_chimera : function
            A function mapping a source node to a chimera coordinate
        nice_to_target: function
            A function mapping a pegasus nice-coordinate to a target node
        offset : tuple (int, int, int)
            A triplet of ints representing the t-, y- and x-offset of the
            sublattice.

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Chimera
            graph to the target Pegasus graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.
        
    """
    t_offset, y_offset, x_offset = offset

    def mapping(q):
        y, x, u, k = source_to_chimera(q)
        return nice_to_target((t_offset, y + y_offset, x + x_offset, u, k))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping


class __pegasus_coordinates_cache_dict(dict):
    """An internal-use cached factory for `pegasus_coordinates` objects"""
    def __missing__(self, key):
        self[key] = val = pegasus_coordinates(key)
        return val

_pegasus_coordinates_cache = __pegasus_coordinates_cache_dict()

# a set of manually-computed values to speed up the generation of sublattice
# mappings, used in _pegasus_pegasus_sublattice_mapping
_sublattice_delta_list = (
    ((0, 0, 0), (1, 0, 0), (2, 0, 0)),
    ((1, 1, 0), (2, 1, 0), (0, 0, 1)),
    ((2, 1, 0), (0, 0, 1), (1, 0, 1)),
)

def _pegasus_pegasus_sublattice_mapping(source_to_nice, nice_to_target, offset):
    """Constructs a mapping from a Pegasus graph to a Pegasus graph, via an offset.
    This function is used by pegasus_sublattice_mappings, and serves to 
    construct a closure that is stable under iteration therein.

    Parameters
    ----------
        source_to_nice : function
            A function mapping a source node to a pegasus nice-coordinate
        nice_to_target: function
            A function mapping a pegasus nice-coordinate to a target node.
        offset : tuple (int, int, int)
            A triplet of ints representing the t-, y- and x-offset of the
            sublattice.

    Returns
    -------
        mapping : function
            The function implementing the mapping from the source Pegasus
            graph to the target Pegasus graph.  We store ``offset`` in the
            attribute ``mapping.offset`` for later reconstruction.
        
    """
    t_offset, y_offset, x_offset = offset
    delta = _sublattice_delta_list[t_offset]
    def mapping(q):
        T, Y, X, u, k = source_to_nice(q)
        t, dy, dx = delta[T]
        return nice_to_target((t, Y + dy + y_offset, X + dx + x_offset, u, k))

    # store the offset in the mapping, so the user can reconstruct it
    mapping.offset = offset

    return mapping


def pegasus_sublattice_mappings(source, target, offset_list=None):
    """Yields mappings from a Chimera or Pegasus graph into a Pegasus graph.
    
    A sublattice mapping is a function from nodes of a ``pegasus_graph(m_s)`` or
    ``chimera_graph(m_c, n_c, 4)`` to nodes of a ``pegasus_graph(m_t)`` with
    ``m_s <= m_t`` or ``m_c <= m_t - 1`` and ``n_c <= m_t - 1``.  This is used
    to identify subgraphs of the target Pegasus graphs which are isomorphic to
    the source graph.  However, if the target graph is not of perfect yield,
    these functions do not generally produce isomorphisms (for example, if a 
    node is missing in the target graph, it may still appear in the image of the
    source graph).
    
    Note that we require the tile parameter of Chimera graphs to be 4, and the
    mappings produced are not exhaustive.  The mappings take the form
    
        ``(y, x, u, k) -> (t_offset, y+y_offset, x+x_offset, u, k)``
        
    when the source is a Chimera graph, or
    
        ``(t, y, x, u, k) -> ((t + t_offset)%3, y+y_offset, x+x_offset, u, k)``
    
    when the source is a Pegasus graph; preserving the orientation and tile
    index of nodes.  We use the notation of Chimera coordinates and Pegasus nice 
    coordinates above, but the mapping produced will respect the labelings of
    the source and target graph.  Note, the notation above for Pegasus->Pegasus
    mappings is only suggestive. See _pegasus_pegasus_sublattice_mapping for a
    precise formula.
    
    Academic note: the full group of isomorphisms of a Chimera graph includes 
    mappings which permute tile indices on a per-row and per-column basis, in
    addition to reflections and rotations of the grid of unit tiles where 
    rotations by 90 and 270 degrees induce a change in orientation.  The
    isomorphisms of Pegasus graphs permit the swapping across rows and columns
    of odd couplers, as well as a reflection about the main antidiagonal which
    induces a change in orientation.  The full set of sublattice mappings would
    take those isomorphisms into account; we do not undertake that complexity
    here.

    Parameters
    ----------
        source : NetworkX Graph
            The Chimera or Pegasus graph that nodes are input from
        target : NetworkX Graph
            The Pegasus graph that nodes are output to
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
    if target.graph.get('family') != 'pegasus':
        raise ValueError("source graphs must a Pegasus graph constructed by dwave_networkx.pegasus_graph")

    m_t = target.graph['rows']
    labels_t = target.graph['labels']
    if labels_t == 'int':
        nice_to_target = _pegasus_coordinates_cache[m_t].nice_to_linear
    elif labels_t == 'coordinate':
        nice_to_target = _pegasus_coordinates_cache[m_t].nice_to_pegasus
    elif labels_t == 'nice':
        def nice_to_target(q):
            return q
    else:
        raise ValueError(f"Pegasus node labeling {labels_t} not recognized")

    labels_s = source.graph['labels']    
    if source.graph.get('family') == 'chimera':
        if source.graph['tile'] != 4:
            raise ValueError("Cannot construct sublattice mappings from Chimera to Pegasus unless the Chimera tile parameter is 4")

        m_s = source.graph['rows']
        n_s = source.graph['columns']
        if offset_list is None:
            offset_list = product([0, 1, 2], range(m_t - m_s), range(m_t - n_s))
        if labels_s == 'coordinate':
            def source_to_inner(q):
                return q
        elif labels_s == 'int':
            source_to_inner = _chimera_coordinates_cache[m_s, n_s, 4].linear_to_chimera
        else:
            raise ValueError(f"Chimera node labeling {labels_s} not recognized")

        make_mapping = _chimera_pegasus_sublattice_mapping

    elif source.graph.get('family') == 'pegasus':
        m_s = source.graph['rows']
        if offset_list is None:
            ranges = range(m_t - m_s + 1), range(m_t - m_s), range(m_t - m_s)
            offset_list = ((t, y, x) for t in range(3) for y in ranges[t] for x in ranges[t])

        labels_s = source.graph['labels']
        if labels_s == 'int':
            source_to_inner = _pegasus_coordinates_cache[m_s].linear_to_nice
        elif labels_s == 'coordinate':
            source_to_inner = _pegasus_coordinates_cache[m_s].pegasus_to_nice
        elif labels_s == 'nice':
            def source_to_inner(q):
                return q
        else:
            raise ValueError(f"Pegasus node labeling {labels_s} not recognized")

        make_mapping = _pegasus_pegasus_sublattice_mapping

    else:
        raise ValueError("source graph must be a Chimera graph or Pegasus graph constructed by dwave_networkx.chimera_graph or dwave_networkx.pegasus_graph respectively")

    for offset in offset_list:
        yield make_mapping(source_to_inner, nice_to_target, offset)


def pegasus_torus(m, node_list=None, edge_list=None, 
                  offset_lists=None, offsets_index=None):
    """
    Creates a Pegasus graph modified to allow for periodic boundary conditions and translational invariance.

    Parameters
    ----------
    m : int
        Size parameter for the Pegasus lattice.
        Connectivity of all nodes is :math:`13 + min(m - 1, 2)`
    node_list : iterable (optional, default None)
        Iterable of nodes in the graph. If None, nodes are generated 
        for an undiluted torus calculated from ``m``
        as described below. The node list must describe a subset
        of the torus nodes to be maintained in the graph 
        using the coordinate node labeling scheme.
    edge_list : iterable (optional, default None)
        Iterable of edges in the graph. If None, edges are generated
        for an undiluted torus calculated from ``m``
        as described below. The edge list must describe 
        a subgraph of the torus, using the coordinate node labeling scheme.
    offset_lists : pair of lists, optional (default None)
        Directly controls the offsets. Each list in the pair must have length 12
        and contain even integers.  If ``offset_lists`` is not None, the ``offsets_index``
        parameter must be None.
    offsets_index : int, optional (default None)
        A number between 0 and 7, inclusive, that selects a preconfigured
        set of topological parameters. If both the ``offsets_index`` and
        ``offset_lists`` parameters are None, the ``offsets_index`` parameters is set
        to zero. At least one of these two parameters must be None.

    Returns
    -------
    G : NetworkX Graph
        A Pegasus torus for size parameter :math:`m` using the coordinate labeling system.


    A Pegasus torus is a generalization of the standard Pegasus graph
    whereby degree-fifteen connectivity is maintained, but the boundary
    condition is modified to enforce an additional translational-invariance 
    symmetry [RH]_. Local connectivity in the Pegasus torus
    is identical to connectivity for Pegasus graph nodes away from the boundary.
    A tile consists of 24 nodes, and the torus has :math:`m - 1` by :math:`m - 1` tiles. 
    Tile displacement modulo :math:`m - 1` defines an automorphism.
    
    See :func:`.pegasus_graph` for additional information.

    Examples
    ========
    >>> G = dnx.pegasus_torus(4)  # a 3x3 tile pegasus torus (connectivity 15)
    >>> len(G) # 3*3*24
    216
    >>> any([len(list(G.neighbors(n))) != 15 for n in G.nodes])
    False

    """
    # It is useful to inherit properties, attributes and methods of G:
    G = pegasus_graph(m=m, node_list=None, edge_list=None, data=True, 
                      coordinates=True, 
                      offset_lists=offset_lists, offsets_index=offsets_index)
    if m<2:
        raise ValueError("m>=2 to define a non-empty lattice")
    # Create the graph minor by contraction of boundary variables
    # (u, m - 1, k, z) to (u, 0, k, z) and match boundary coupling to the
    # bulk with addition of supplementary external couplers 
    def relabel(u,w,k,z):
        return (u, w%(m - 1), k, z)

    # Contract internal couplers spanning the boundary:
    G.add_edges_from([(relabel(*edge[0]), relabel(*edge[1]))
                      for edge in G.edges() if edge[0][1]==m - 1 or edge[1][1]==m - 1])
    if m>3:
        # Add missing external couplers  (u, w, k, -1) and (u, w, k, 0). 
        G.add_edges_from([((u, w, k, m - 2), (u, w, k, 0))
                          for u in range(2)
                          for w in range(m - 1)
                          for k in range(12)])
    else:
        # 2-tile wide lattices do not allow for boundary spanning
        # edges.
        pass
    # Delete variables contracted at the boundary:
    G.remove_nodes_from([(u, m - 1, k, z)
                         for u in range(2) for k in range(12) for z in range(m - 1)])
    _add_compatible_terms(G, node_list, edge_list)

    G.graph['boundary_condition'] = 'torus'
    
    return G
