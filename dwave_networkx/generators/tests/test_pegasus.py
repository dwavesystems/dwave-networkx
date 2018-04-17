from __future__ import division

import unittest

import networkx as nx
import dwave_networkx as dnx

alpha_map = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))


class TestPegasusGraph(unittest.TestCase):
    def test_p2(self):
        G = dnx.pegasus_graph(2, fabric_only=False)
    
        # should have 48 nodes
        self.assertEqual(len(G), 48)

        # nodes 0,...,47 should be in the graph
        for n in range(48):
            self.assertIn(n, G)

    def test_connected_component(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        test_offsets = [[0] * 12] * 2, [[2] * 12, [6] * 12], [[6] * 12, [2, 6, 10] * 4], [[2, 6, 10] * 4] * 2
        for offsets in test_offsets:
            G = dnx.pegasus_graph(4, fabric_only=True, offset_lists=offsets)
            H = dnx.pegasus_graph(4, fabric_only=False, offset_lists=offsets)
            nodes = sorted(G)
            comp = sorted(max(nx.connected_components(H), key=len))
            self.assertEqual(comp, nodes)

    def test_coordinate_basics(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        G = dnx.pegasus_graph(4, fabric_only=False)
        H = dnx.pegasus_graph(4, coordinates=True, fabric_only=False)
        coords = pegasus_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['pegasus_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

    def test_coordinate_subgraphs(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        from random import sample
        G = dnx.pegasus_graph(4)
        H = dnx.pegasus_graph(4, coordinates=True)
        coords = pegasus_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.tuples(lmask))

        self.assertEqual(lmask, list(coords.ints(cmask)))

        Gm = dnx.pegasus_graph(4, node_list=lmask)
        Hm = dnx.pegasus_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.pegasus_graph(4, edge_list=EG)
        Hn = dnx.pegasus_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['pegasus_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

        self.assertEqual(EG, sorted(map(sorted, coords.int_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.tuple_pairs(Gn.edges()))))
