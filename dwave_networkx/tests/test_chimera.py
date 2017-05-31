import unittest

import dwave_networkx as dnx


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
