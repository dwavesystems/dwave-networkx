"""
Generators for some graphs derived from the D-Wave System.

"""
import networkx as nx

from dwave_networkx import _PY2
from dwave_networkx.exceptions import DWaveNetworkXException

__all__ = ['pegasus_graph']

# compatibility for python 2/3
if _PY2:
    range = xrange


def pegasus_graph(m, create_using=None, node_list=None, edge_list=None, data=True, offset_lists=None, offsets_index=None, coordinates=False):
    """
    Creates a Pegasus graph of size (m, m).
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
    else:
        if offsets_index is not None:
            raise DWaveNetworkXException, "provide at most one of offsets_index and offset_lists"
        offsets_descriptor = offset_lists

    G = nx.empty_graph(0, create_using)

    G.name = "pegasus_graph(%s, %s)" % (m, offsets_descriptor)

    max_size = m * (m - 1) * 24  # max number of nodes G can have

    m1 = m - 1
    if coordinates:
        c2i = lambda *q: q
    else:
        def c2i(u, w, k, z): return u * 12 * m * m1 + w * 12 * m1 + k * m1 + z

    if edge_list is None:
        G.add_edges_from((c2i(u, w, k, z), c2i(u, w, k, z + 1))
                         for u in (0, 1)
                         for w in range(m)
                         for k in range(12)
                         for z in range(m1 - 1))

        G.add_edges_from((c2i(u, w, 2 * k, z), c2i(u, w, 2 * k + 1, z))
                         for u in (0, 1)
                         for w in range(m)
                         for k in range(6)
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
