from __future__ import division

import unittest

import networkx as nx
import dwave_networkx as dnx

try:
    import matplotlib.pyplot as plt
    _plt = True
except ImportError:
    _plt = False

try:
    import numpy as np
    _numpy = True
except ImportError:
    _numpy = False


class TestDrawing(unittest.TestCase):
    @unittest.skipIf(not _numpy, "No numpy")
    def test_chimera_layout_basic(self):
        G = dnx.chimera_graph(1, 1, 4)
        pos = dnx.chimera_layout(G)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_chimera_layout_typical(self):
        G = dnx.chimera_graph(2, 2, 4)
        pos = dnx.chimera_layout(G)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_chimera_layout_no_chimera_indices(self):
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 2), (1, 3), (0, 3)])
        pos = dnx.chimera_layout(G)
        pos2 = dnx.chimera_layout(dnx.chimera_graph(1, 1, 2))

        for v in pos:
            self.assertTrue(all(pos[v] == pos2[v]))
        for v in pos2:
            self.assertIn(v, pos)

    # def test_draw_chimera_typical(self):
    #     G = dnx.chimera_graph(5, 5, 3)

    #     h = {v: v % 2 and v / len(G) or v * -1 / len(G) for v in G}
    #     J = {(u, v): h[v] * h[u] for u, v in G.edges_iter()}
    #     dnx.draw_chimera(G, linear_biases=h, quadratic_biases=J, with_labels=True)
    #     plt.savefig('test.ps')
