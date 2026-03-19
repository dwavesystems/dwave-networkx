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

import os
import unittest

import networkx as nx
import dwave.graphs as dnx

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

try:
    import numpy as np
except ImportError:
    np = False

_display = os.environ.get('DISPLAY', '') != ''

@unittest.skipUnless(np and plt, "No numpy or matplotlib")
class TestDrawing(unittest.TestCase):
    def test_zephyr_layout_basic(self):
        G = dnx.zephyr_graph(1, 4)
        pos = dnx.zephyr_layout(G)

    def test_zephyr_layout_typical(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G)

    def test_zephyr_layout_center(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G, center=(5, 5))
        with self.assertRaises(ValueError):
            pos = dnx.zephyr_layout(G, center=(5, 5, 5))

    def test_zephyr_layout_lowdim(self):
        G = dnx.zephyr_graph(2, 4)
        with self.assertRaises(ValueError):
            pos = dnx.zephyr_layout(G, dim=1)

    def test_zephyr_layout_weird_nodata(self):
        G = dnx.zephyr_graph(2, 4)
        del G.graph["family"]
        with self.assertRaises(ValueError):
            pos = dnx.zephyr_layout(G, dim=1)

    def test_zephyr_layout_no_zephyr_indices(self):
        pos = dnx.zephyr_layout(dnx.zephyr_graph(1, 1, data=False))
        pos2 = dnx.zephyr_layout(dnx.zephyr_graph(1, 1))
        coord = dnx.zephyr_coordinates(1, 1)
        for v in pos:
            self.assertTrue(all(pos[v] == pos2[v]))
        for v in pos2:
            self.assertIn(v, pos)

    def test_zephyr_layout_coords(self):
        G = dnx.zephyr_graph(2, 4, coordinates=True)
        pos = dnx.zephyr_layout(G)

    def test_zephyr_layout_nodata(self):
        G = dnx.zephyr_graph(2, 4, data=False)
        pos = dnx.zephyr_layout(G)
        
    def test_zephyr_layout_xrange_typical(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G)
        x_coords = [val[0] for val in pos.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        self.assertAlmostEqual(min_x, 0, delta=1e-5, msg="min_x should be approximately 0")
        self.assertAlmostEqual(max_x, 1, delta=1e-5, msg="max_x should be approximately 1")
        
    def test_zephyr_layout_yrange_typical(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G)
        y_coords = [val[1] for val in pos.values()]
        min_y, max_y = min(y_coords), max(y_coords)
        self.assertAlmostEqual(min_y, -1, delta=1e-5, msg="min_y should be approximately -1")
        self.assertAlmostEqual(max_y, 0, delta=1e-5, msg="max_y should be approximately 0")

    def test_zephyr_layout_xrange(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G, scale=10)
        x_coords = [val[0] for val in pos.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        self.assertAlmostEqual(min_x, 0, delta=1e-5, msg="min_x should be approximately 0")
        self.assertAlmostEqual(max_x, 10, delta=1e-5, msg="max_x should be approximately 10")
        
    def test_zephyr_layout_yrange(self):
        G = dnx.zephyr_graph(2, 4)
        pos = dnx.zephyr_layout(G, scale=10)
        y_coords = [val[1] for val in pos.values()]
        min_y, max_y = min(y_coords), max(y_coords)
        self.assertAlmostEqual(min_y, -10, delta=1e-5, msg="min_y should be approximately -10")
        self.assertAlmostEqual(max_y, 0, delta=1e-5, msg="max_y should be approximately 0")

    @unittest.skipUnless(_display, " No display found")
    def test_draw_zephyr_yield(self):
        G = dnx.zephyr_graph(2, 4, data=False)
        G.remove_edges_from([(51, 159),(51, 159),(24, 98)])
        G.remove_nodes_from([18,23])
        dnx.draw_zephyr_yield(G)

    @unittest.skipUnless(_display, " No display found")
    def test_draw_zephyr_biases(self):
        G = dnx.zephyr_graph(8)
        h = {v: v % 12 for v in G}
        J = {(u, v) if u % 2 else (v, u): (u+v) % 24 for u, v in G.edges()}
        for v in G:
            J[v, v] = .1

        dnx.draw_zephyr(G, linear_biases=h, quadratic_biases=J)

    @unittest.skipUnless(_display, " No display found")
    def test_draw_zephyr_embedding(self):
        C = dnx.zephyr_graph(4)
        G = nx.grid_graph([2, 3, 2])
        emb = {(0, 0, 0): [80, 48], (0, 0, 1): [50, 52], (0, 1, 0): [85, 93],
               (0, 1, 1): [84, 82], (0, 2, 0): [89], (0, 2, 1): [92],
               (1, 0, 0): [49, 54], (1, 0, 1): [83, 51], (1, 1, 0): [81],
               (1, 1, 1): [86, 94], (1, 2, 0): [87, 95], (1, 2, 1): [91]}
        dnx.draw_zephyr_embedding(C, emb)
        dnx.draw_zephyr_embedding(C, emb, embedded_graph=G)
        dnx.draw_zephyr_embedding(C, emb, interaction_edges=C.edges())

    @unittest.skipUnless(_display, " No display found")
    def test_draw_overlapped_zephyr_embedding(self):
        C = dnx.zephyr_graph(2)
        emb = {0: [1, 3], 1: [3, 128, 15], 2: [25, 140], 3: [17, 122]}
        dnx.draw_zephyr_embedding(C, emb, overlapped_embedding=True)
        dnx.draw_zephyr_embedding(C, emb, overlapped_embedding=True, show_labels=True)
