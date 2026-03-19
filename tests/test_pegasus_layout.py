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
    def test_pegasus_layout_coords(self):
        G = dnx.pegasus_graph(2, coordinates=True)
        pos = dnx.pegasus_layout(G)

    def test_pegasus_layout_ints(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G)

    def test_pegasus_layout_chim(self):
        G = dnx.pegasus_graph(2, nice_coordinates=True)
        pos = dnx.pegasus_layout(G)

    def test_pegasus_layout_ints_nodata(self):
        G = dnx.pegasus_graph(2, data=False)
        pos = dnx.pegasus_layout(G)

    def test_pegasus_layout_crosses(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G, crosses=True)

    def test_pegasus_layout_ints_badcenter(self):
        G = dnx.pegasus_graph(2, data=False)
        with self.assertRaises(ValueError):
            pos = dnx.pegasus_layout(G, center=(0, 0, 0, 0))

    def test_pegasus_layout_ints_noinfo(self):
        G = dnx.pegasus_graph(2, data=False)
        badG = nx.Graph()
        badG.add_edges_from(G.edges())
        with self.assertRaises(ValueError):
            pos = dnx.pegasus_layout(badG)

    def test_pegasus_layout_xrange_typical(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G)
        x_coords = [val[0] for val in pos.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        self.assertAlmostEqual(min_x, 0, delta=1e-5, msg="min_x should be approximately 0")
        self.assertAlmostEqual(max_x, 1, delta=1e-5, msg="max_x should be approximately 1")
        
    def test_pegasus_layout_yrange_typical(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G)
        y_coords = [val[1] for val in pos.values()]
        min_y, max_y = min(y_coords), max(y_coords)
        self.assertAlmostEqual(min_y, -1, delta=1e-5, msg="min_y should be approximately -1")
        self.assertAlmostEqual(max_y, 0, delta=1e-5, msg="max_y should be approximately 0")

    def test_pegasus_layout_xrange(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G, scale=5)
        x_coords = [val[0] for val in pos.values()]
        min_x, max_x = min(x_coords), max(x_coords)
        self.assertAlmostEqual(min_x, 0, delta=1e-5, msg="min_x should be approximately 0")
        self.assertAlmostEqual(max_x, 5, delta=1e-5, msg="max_x should be approximately 5")
        
    def test_pegasus_layout_yrange(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G, scale=5)
        y_coords = [val[1] for val in pos.values()]
        min_y, max_y = min(y_coords), max(y_coords)
        self.assertAlmostEqual(min_y, -5, delta=1e-5, msg="min_y should be approximately -5")
        self.assertAlmostEqual(max_y, 0, delta=1e-5, msg="max_y should be approximately 0")

    @unittest.skipUnless(_display, " No display found")
    def test_draw_pegasus_yield(self):
        G = dnx.pegasus_graph(3, data=False)
        G.remove_edges_from([(5,104),(12,96),(23,112)])
        G.remove_nodes_from([109,139])
        dnx.draw_pegasus_yield(G)

    @unittest.skipUnless(_display, " No display found")
    def test_draw_pegasus_biases(self):
        G = dnx.pegasus_graph(2)
        h = {v: v % 12 for v in G}
        J = {(u, v) if u % 2 else (v, u): (u+v) % 24 for u, v in G.edges()}
        for v in G:
            J[v, v] = .1

        dnx.draw_pegasus(G, linear_biases=h, quadratic_biases=J)

    @unittest.skipUnless(_display, " No display found")
    def test_draw_pegasus_embedding(self):
        P = dnx.pegasus_graph(2)
        G = nx.grid_graph([3, 3, 2])
        emb = {(0, 0, 0): [35], (0, 0, 1): [12], (0, 0, 2): [31], (0, 1, 0): [16],
               (0, 1, 1): [36], (0, 1, 2): [11], (0, 2, 0): [39], (0, 2, 1): [6],
               (0, 2, 2): [41], (1, 0, 0): [34], (1, 0, 1): [13], (1, 0, 2): [30],
               (1, 1, 0): [17], (1, 1, 1): [37], (1, 1, 2): [10], (1, 2, 0): [38],
               (1, 2, 1): [7], (1, 2, 2): [40]}
        dnx.draw_pegasus_embedding(P, emb)
        dnx.draw_pegasus_embedding(P, emb, embedded_graph=G)
        dnx.draw_pegasus_embedding(P, emb, interaction_edges=P.edges())
        dnx.draw_pegasus_embedding(P, emb, crosses=True)

    @unittest.skipUnless(_display, " No display found")
    def test_draw_overlapped_chimera_embedding(self):
        C = dnx.pegasus_graph(2)
        emb = {0: [12, 35], 1: [12, 31], 2: [32], 3: [14]}
        dnx.draw_pegasus_embedding(C, emb, overlapped_embedding=True)
        dnx.draw_pegasus_embedding(C, emb, overlapped_embedding=True, show_labels=True)
