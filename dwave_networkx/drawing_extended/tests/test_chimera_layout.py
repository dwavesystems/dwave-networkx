from __future__ import division

import unittest

import matplotlib.pyplot as plt
import dwave_networkx as dnx
from dwave_networkx.drawing_extended.chimera_layout import _find_chimera_indices


class TestDrawing(unittest.TestCase):
    def test__find_chimera_indices_one_tile_typical(self):
        for L in range(1, 10):
            G = dnx.chimera_graph(1, 1, L)
            chimera_indices = _find_chimera_indices(G)

            for v, dat in G.nodes(data=True):
                self.assertEqual(dat['chimera_index'], chimera_indices[v])

    def test__find_chimera_indices_one_tile_degenerate(self):
        G = dnx.chimera_graph(1, 1, 5)

        # remove 1 node
        G.remove_node(4)
        chimera_indices = _find_chimera_indices(G)
        for v, dat in G.nodes(data=True):
            self.assertEqual(dat['chimera_index'], chimera_indices[v])

        # remove another node
        G.remove_node(3)
        chimera_indices = _find_chimera_indices(G)
        for v, dat in G.nodes(data=True):
            self.assertEqual(dat['chimera_index'], chimera_indices[v])

    # def test__find_chimera_indices_typical(self):

    #     G = dnx.chimera_graph(2, 2, 4)
    #     chimera_indices = _find_chimera_indices(G)

    def test_chimera_layout_basic(self):
        # TODO
        G = dnx.chimera_graph(1, 1, 4)

        pos = dnx.chimera_layout(G)

        print(pos)

    def test_chimera_layout_typical(self):
        # TODO
        G = dnx.chimera_graph(2, 2, 4)
        pos = dnx.chimera_layout(G)

        print(pos)

    def test_draw_chimera_typical(self):
        G = dnx.chimera_graph(5, 5, 3)

        h = {v: v % 2 and v / len(G) or v * -1 / len(G) for v in G}
        J = {(u, v): h[v] * h[u] for u, v in G.edges_iter()}
        dnx.draw_chimera(G, linear_biases=h, quadratic_biases=J, with_labels=True)
        plt.show()
