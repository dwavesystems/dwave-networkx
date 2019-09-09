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
from __future__ import division

import unittest

import networkx as nx
import dwave_networkx as dnx

alpha_map = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))


class TestChimeraGraph(unittest.TestCase):
    def test_single_tile(self):

        # fully specified
        G = dnx.chimera_graph(1, 1, 4)

        # should have 8 nodes
        self.assertEqual(len(G), 8)

        # nodes 0,...,7 should be in the tile
        for n in range(8):
            self.assertIn(n, G)

        # check bipartite
        for i in range(4):
            for j in range(4, 8):
                self.assertTrue((i, j) in G.edges() or (j, i) in G.edges())

    def test_find_chimera_indices_single_tile(self):
        for k in range(1, 10):
            G = dnx.chimera_graph(1, 1, k)

            # get the chimera indices, check that they match the ones already
            # present
            chimera_indices = dnx.find_chimera_indices(G)
            self._check_matching_chimera_indices(G, chimera_indices)

    def test_find_chimera_indices_single_tile_alpha_labels(self):
        for k in range(1, 10):
            G = nx.relabel_nodes(dnx.chimera_graph(1, 1, k), alpha_map)

            # get the chimera indices, check that they match the ones already
            # present
            chimera_indices = dnx.find_chimera_indices(G)
            self._check_matching_chimera_indices(G, chimera_indices)

    def test_find_chimera_indices_one_tile_degenerate(self):
        G = dnx.chimera_graph(1, 1, 5)

        # remove 1 node
        G.remove_node(4)
        chimera_indices = dnx.find_chimera_indices(G)
        self._check_matching_chimera_indices(G, chimera_indices)

        # remove another node
        G.remove_node(3)
        chimera_indices = dnx.find_chimera_indices(G)
        self._check_matching_chimera_indices(G, chimera_indices)

    def test_not_full_yield(self):
        edges = [(0, 3), (0, 2), (1, 3)]
        G = dnx.chimera_graph(1, 1, 2, edge_list=edges)
        for e in edges:
            self.assertIn(e, G.edges())
        for (u, v) in G.edges:
            self.assertTrue((u, v) in edges or (v, u) in edges)

        nodes = [0, 1, 2]
        G = dnx.chimera_graph(1, 1, 2, node_list=nodes)
        self.assertTrue(len(G) == 3)
        self.assertTrue(len(G.edges()) == 2)

        edges = [(0, 2), (1, 2)]
        nodes = [0, 1, 2, 3]
        G = dnx.chimera_graph(1, 1, 2, node_list=nodes, edge_list=edges)
        # 3 should be added as a singleton
        self.assertTrue(len(G[3]) == 0)

    def test_float_robustness(self):
        G = dnx.chimera_graph(8 / 2)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.chimera_graph(4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.chimera_graph(4, 4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

    # def test_find_chimera_indices_typical(self):
    #     for t in range(2, 5):
    #         G = dnx.chimera_graph(2, 2, t)
    #         chimera_indices = dnx.find_chimera_indices(G)
    #         self._check_matching_chimera_indices(G, chimera_indices)

    #         G = dnx.chimera_graph(4, 4, t)
    #         chimera_indices = dnx.find_chimera_indices(G)
    #         self._check_matching_chimera_indices(G, chimera_indices)

    # def test_find_chimera_indices_shore_1(self):
    #     G = dnx.chimera_graph(2, 2, 1)
    #     chimera_indices = dnx.find_chimera_indices(G)
    #     self._check_matching_chimera_indices(G, chimera_indices)

    #     G = dnx.chimera_graph(4, 4, 1)
    #     chimera_indices = dnx.find_chimera_indices(G)
    #     self._check_matching_chimera_indices(G, chimera_indices)

    def _check_matching_chimera_indices(self, G, chimera_indices):
        for v, dat in G.nodes(data=True):
            self.assertEqual(dat['chimera_index'], chimera_indices[v])

    def test_coordinate_basics(self):
        from dwave_networkx.generators.chimera import chimera_coordinates
        G = dnx.chimera_graph(4)
        H = dnx.chimera_graph(4, coordinates=True)
        coords = chimera_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['chimera_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['chimera_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

    def test_coordinate_subgraphs(self):
        from dwave_networkx.generators.chimera import chimera_coordinates
        from random import sample
        G = dnx.chimera_graph(4)
        H = dnx.chimera_graph(4, coordinates=True)
        coords = chimera_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.tuples(lmask))

        self.assertEqual(lmask, list(coords.ints(cmask)))

        Gm = dnx.chimera_graph(4, node_list=lmask)
        Hm = dnx.chimera_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.chimera_graph(4, edge_list=EG)
        Hn = dnx.chimera_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['chimera_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['chimera_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

        self.assertEqual(EG, sorted(map(sorted, coords.int_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.tuple_pairs(Gn.edges()))))

    def test_linear_to_chimera(self):
        G = dnx.linear_to_chimera(212, 8, 8, 4)
        self.assertEqual(G, (3, 2, 1, 0))

    def test_chimera_to_linear(self):
        G = dnx.chimera_to_linear(3, 2, 1, 0, 8, 8, 4)
        self.assertEqual(G, 212)
