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

import unittest

import networkx as nx
import dwave_networkx as dnx

class TestZephyrGraph(unittest.TestCase):
    def test_single_tile(self):

        # fully specified
        G = dnx.zephyr_graph(1, 4)

        # should have 8 nodes
        self.assertEqual(len(G), 48)

        # nodes 0,...,7 should be in the tile
        for n in range(48):
            self.assertIn(n, G)


    def test_not_full_yield(self):
        edges =  [(2, 30), (7, 44), (10, 37), (12, 29), (15, 37), (19, 41)]
        G = dnx.zephyr_graph(1, 4, edge_list=edges)
        for e in edges:
            self.assertIn(e, G.edges())
        for (u, v) in G.edges:
            self.assertTrue((u, v) in edges or (v, u) in edges)

        nodes = [0, 1, 2]
        G = dnx.zephyr_graph(1, 2, node_list=nodes)
        self.assertTrue(len(G) == 3)
        self.assertTrue(len(G.edges()) == 1)

        edges = [(0, 1), (2, 3)]
        nodes = [0, 1, 2, 3]
        G = dnx.zephyr_graph(1, 2, node_list=nodes, edge_list=edges)
        self.assertTrue(len(G) == 4)
        self.assertTrue(len(G.edges()) == 2)

    def test_float_robustness(self):
        G = dnx.zephyr_graph(8 / 2)

        self.assertEqual(set(G.nodes), set(dnx.zephyr_graph(4).nodes))
        for u, v in dnx.zephyr_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.zephyr_graph(4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.zephyr_graph(4).nodes))
        for u, v in dnx.zephyr_graph(4).edges:
            self.assertIn(u, G[v])

    def _check_matching_zephyr_indices(self, G, zephyr_indices):
        for v, dat in G.nodes(data=True):
            self.assertEqual(dat['zephyr_index'], zephyr_indices[v])

    def test_coordinate_basics(self):
        from dwave_networkx.generators.zephyr import zephyr_coordinates
        G = dnx.zephyr_graph(4)
        H = dnx.zephyr_graph(4, coordinates=True)
        coords = zephyr_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['zephyr_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['zephyr_index'], q)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))

    def test_coordinate_subgraphs(self):
        from dwave_networkx.generators.zephyr import zephyr_coordinates
        from random import sample
        G = dnx.zephyr_graph(4)
        H = dnx.zephyr_graph(4, coordinates=True)
        coords = zephyr_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.iter_linear_to_zephyr(lmask))

        self.assertEqual(lmask, list(coords.iter_zephyr_to_linear(cmask)))

        Gm = dnx.zephyr_graph(4, node_list=lmask)
        Hm = dnx.zephyr_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.zephyr_graph(4, edge_list=EG)
        Hn = dnx.zephyr_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['zephyr_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['zephyr_index'], q)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))

        self.assertEqual(EG, sorted(map(sorted, coords.iter_zephyr_to_linear_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.iter_linear_to_zephyr_pairs(Gn.edges()))))
