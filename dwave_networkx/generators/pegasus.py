"""
Generators for some graphs derived from the D-Wave System.

"""
import networkx as nx

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException
import warnings

__all__ = ['pegasus_graph']

# compatibility for python 2/3
if _PY2:
    range = xrange


def pegasus_graph(m, create_using=None, node_list=None, edge_list=None, data=True, offset_lists=None, offsets_index=None, coordinates=False, fabric_only=True):
    """
    Creates a Pegasus graph with size parameter m.  The Pegasus topology produced
    by this generator with default parameters is one member of a large family of
    topologies under consideration, and may not be reflected in future products.

    A Pegasus lattice is a graph minor of a lattice similar to Chimera,
    where unit tiles are completely connected.  In the most generality, our
    prelattice Q(N0,N1) contains nodes of the form
        (i, j, 0, k) with 0 <= k < 2 [vertical nodes]
    and
        (i, j, 1, k) with 0 <= k < 2 [horizontal nodes]
    for 0 <= i <= N0 and 0 <= j < N1; and edges of the form
        (i, j, u, k) ~ (i+u, j+1-u, u, k)  [external edges]
        (i, j, 0, k) ~ (i, j, 1, k) [internal edges]
        (i, j, u, 0) ~ (i, j, u, 1) [odd edges]

    The minor is specified by two lists of offsets; S0 and S1 of length L0 and L1
    (where L0 and L1, and the entries of S0 and S1, must be divisible by 2).
    From these offsets, we construct our minor, a Pegasus lattice, by contracting
    the complete intervals of external edges,
        I(0, w, k, z) = [(L1*w + k, L0*z + S0[k] + r, 0, k % 2) for 0 <= r < L0]
        I(1, w, k, z) = [(L1*z + S1[k] + r, L0*w + k, 1, k % 2) for 0 <= r < L1]
    and deleting the prelattice nodes of any interval not fully contained in
    Q(N0, N1).

    This generator is specialized to L0 = L1 = 12; N0 = N1 = 12m.

    The notation (u, w, k, z) is called the pegasus index of a node in a pegasus
    lattice.  The entries can be interpreted as following,
        u : qubit orientation (0 = vertical, 1 = horizontal)
        w : orthogonal major offset
        k : orthogonal minor offset
        z : parallel offset
    and the edges in the minor have the form
        (u, w, k, z) ~ (u, w, k, z+1) [external edges]
        (0, w0, k0, z0) ~ (1, w1, k1, z1) [internal edges, see below]
        (u, w, 2k, z) ~ (u, w, 2k+1, z) [odd edges]
    where internal edges only exist when
        w1 = z0 + (1 if k1 < S0[k0] else 0), and
        z1 = w0 - (1 if k0 < S1[k1] else 0)

    linear indices are computed from pegasus indices by the formula
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
        If True, node labels are 4-tuple Pegasus indices
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
        constructed.  Ignored if edge_lists is not None
    """
    warnings.warn("The Pegasus topology produced by this generator with default parameters is one member of a large family of topologies under consideration, and may not be reflected in future products")

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

    construction = (("family", "pegasus"), ("rows", m), ("columns", m),
                    ("tile", 12), ("vertical_offsets", offset_lists[0]),
                    ("horizontal_offsets", offset_lists[1]), ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    G.graph.update(construction)

    max_size = m * (m - 1) * 24  # max number of nodes G can have

    m1 = m - 1
    if coordinates:
        c2i = lambda *q: q
    else:
        def c2i(u, w, k, z): return u * 12 * m * m1 + w * 12 * m1 + k * m1 + z

    if edge_list is None:
        if fabric_only:
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
        G.add_edges_from((c2i(0, w, k, z), c2i(1, z + (kk < off0[k]), kk, w - (k < off1[kk])))
                         for w in range(m)
                         for kk in range(12)
                         for k in range(0 if w else off1[kk], 12 if w < m1 else off1[kk])
                         for z in range(m1))

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
                        if coordinates:
                            if q in G:
                                G.node[q]['linear_index'] = v
                        else:
                            if v in G:
                                G.node[v]['pegasus_index'] = q
                        v += 1

    return G


def pegasus_elimination_order(n, coordinates=False):
    '''
    Produces a variable elimination order for a pegasus P(n) graph, which provides an upper bound on the treewidth.

    Simple pegasus variable elimination order rules:
       - eliminate vertical qubits, one column at a time
       - eliminate horizontal qubits in each column once their adjacent vertical qubits have been eliminated

    Many other orderings are possible giving the same treewidth upper bound of 12n-4 for a P(n)
    see pegasus_var_order_extra.py for a few.
    P(n) contains cliques of size 12n-10 so we know the treewidth of P(n) is in [12n-11,12n-4].

    the treewidth bound generated by a variable elimination order can be verified using
    dwave_networkx.elimination_order_width. Example:
    import dwave_networkx as dwnx
    n = 6
    P = dwnx.generators.pegasus_graph(n)
    order = pegasus_var_order(n)
    print(dwnx.elimination_order_width(P,order))
    '''
    m = n
    l = 12

    # ordering for horizontal qubits in each tile, from east to west:
    h_order = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]
    order = []
    for n_i in range(n):  # for each tile offset
        # eliminate vertical qubits:
        for l_i in range(0, l, 2):
            for l_v in range(l_i, l_i + 2):
                for m_i in range(m - 1):  # for each column
                    order.append((0, n_i, l_v, m_i))
            # eliminate horizontal qubits:
            if n_i > 0 and not(l_i % 4):
                # a new set of horizontal qubits have had all their neighbouring vertical qubits eliminated.
                for m_i in range(m):
                    for l_h in range(h_order[l_i], h_order[l_i] + 4):
                        order.append((1, m_i, l_h, n_i - 1))

    if coordinates:
        return order
    else:
        return pegasus_coordinates(n).ints(order)

# i acknowledge that this code duplication is silly but at least it's fast
class pegasus_coordinates:
    def __init__(self, m):
        """
        Provides coordinate converters for the pegasus indexing scheme.

        Parameters
        ----------
        m : int
            The size parameter for the Pegasus lattice.
        """

        self.args = m, m - 1

    def int(self, q):
        """
        Converts the chimera_index `q` into an linear_index

        Parameters
        ----------
        q : tuple
            The chimera_index node label    

        Returns
        -------
        r : int
            The linear_index node label corresponding to q            
        """

        u, w, k, z = q
        m, m1 = self.args
        return ((m * u + w) * 12 + k) * m1 + z

    def tuple(self, r):
        """
        Converts the linear_index `q` into an pegasus_index

        Parameters
        ----------
        r : int
            The linear_index node label    

        Returns
        -------
        q : tuple
            The pegasus_index node label corresponding to r
        """

        m, m1 = self.args
        r, z = divmod(r, m1)
        r, k = divmod(r, 12)
        u, w = divmod(r, m)
        return u, w, k, z

    def ints(self, qlist):
        """
        Converts a sequence of pegasus_index node labels into
        linear_index node labels, preserving order

        Parameters
        ----------
        qlist : sequence of ints
            The pegasus_index node labels

        Returns
        -------
        rlist : iterable of tuples
            The linear_lindex node lables corresponding to qlist
        """

        m, m1 = self.args
        return (((m * u + w) * 12 + k) * m1 + z for (u, w, k, z) in qlist)

    def tuples(self, rlist):
        """
        Converts a sequence of linear_index node labels into
        pegasus_index node labels, preserving order

        Parameters
        ----------
        rlist : sequence of tuples
            The linear_index node labels

        Returns
        -------
        qlist : iterable of ints
            The pegasus_index node lables corresponding to rlist
        """

        m, m1 = self.args
        for r in rlist:
            r, z = divmod(r, m1)
            r, k = divmod(r, 12)
            u, w = divmod(r, m)
            yield u, w, k, z

    def __pair_repack(self, f, plist):
        """
        Flattens a sequence of pairs to pass through `f`, and then
        re-pairs the result.

        Parameters
        ----------
        f : callable
            A function that accepts a sequence and returns a sequence
        plist:
            A sequence of pairs

        Returns
        -------
        qlist : sequence
            Equivalent to (tuple(f(p)) for p in plist)
        """
        ulist = f(u for p in plist for u in p)
        for u in ulist:
            v = next(ulist)
            yield u, v

    def int_pairs(self, plist):
        """
        Translates a sequence of pairs of chimera_index tuples
        into a a sequence of pairs of linear_index ints.

        Parameters
        ----------
        plist:
            A sequence of pairs of tuples

        Returns
        -------
        qlist : sequence
            Equivalent to (tuple(self.ints(p)) for p in plist)
        """
        return self.__pair_repack(self.ints, plist)

    def tuple_pairs(self, plist):
        """
        Translates a sequence of pairs of chimera_index tuples
        into a a sequence of pairs of linear_index ints.

        Parameters
        ----------
        plist:
            A sequence of pairs of tuples

        Returns
        -------
        qlist : sequence
            Equivalent to (tuple(self.tuples(p)) for p in plist)
        """
        return self.__pair_repack(self.tuples, plist)
