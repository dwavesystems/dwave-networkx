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
