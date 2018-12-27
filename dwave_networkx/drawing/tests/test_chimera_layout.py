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

import os
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

if os.environ.get('DISPLAY', '') == '':
    _display = False
else:
    _display = True


class TestDrawing(unittest.TestCase):
    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_basic(self):
        G = dnx.chimera_graph(1, 1, 4)
        pos = dnx.chimera_layout(G)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_typical(self):
        G = dnx.chimera_graph(2, 2, 4)
        pos = dnx.chimera_layout(G)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_center(self):
        G = dnx.chimera_graph(2, 2, 4)
        pos = dnx.chimera_layout(G, center=(5, 5))
        with self.assertRaises(ValueError):
            pos = dnx.chimera_layout(G, center=(5, 5, 5))

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_lowdim(self):
        G = dnx.chimera_graph(2, 2, 4)
        with self.assertRaises(ValueError):
            pos = dnx.chimera_layout(G, dim=1)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_weird_nodata(self):
        G = dnx.chimera_graph(2, 2, 4)
        del G.graph["family"]
        with self.assertRaises(ValueError):
            pos = dnx.chimera_layout(G, dim=1)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_no_chimera_indices(self):
        G = nx.Graph()
        G.add_edges_from([(0, 2), (1, 2), (1, 3), (0, 3)])
        pos = dnx.chimera_layout(G)
        pos2 = dnx.chimera_layout(dnx.chimera_graph(1, 1, 2))

        for v in pos:
            self.assertTrue(all(pos[v] == pos2[v]))
        for v in pos2:
            self.assertIn(v, pos)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_coords(self):
        G = dnx.chimera_graph(2, 2, 4, coordinates=True)
        pos = dnx.chimera_layout(G)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_nodata(self):
        G = dnx.chimera_graph(2, 2, 4, data=False)
        pos = dnx.chimera_layout(G)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    def test_chimera_layout_edgelist_singletile(self):
        G = dnx.chimera_graph(1, 1, 16, data=False)
        pos = dnx.chimera_layout(G.edges())


    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    @unittest.skipUnless(_display, " No display found")
    def test_draw_pegasus_biases(self):
        G = dnx.chimera_graph(8)
        h = {v: v % 12 for v in G}
        J = {(u, v) if u % 2 else (v, u): (u+v) % 24 for u, v in G.edges()}
        for v in G:
            J[v, v] = .1

        dnx.draw_chimera(G, linear_biases=h, quadratic_biases=J)

    @unittest.skipUnless(_numpy and _plt, "No numpy or matplotlib")
    @unittest.skipUnless(_display, " No display found")
    def test_draw_pegasus_embedding(self):
        C = dnx.chimera_graph(4)
        G = nx.grid_graph([2, 3, 2])
        emb = {(0, 0, 0): [80, 48], (0, 0, 1): [50, 52], (0, 1, 0): [85, 93],
               (0, 1, 1): [84, 82], (0, 2, 0): [89], (0, 2, 1): [92],
               (1, 0, 0): [49, 54], (1, 0, 1): [83, 51], (1, 1, 0): [81],
               (1, 1, 1): [86, 94], (1, 2, 0): [87, 95], (1, 2, 1): [91]}
        dnx.draw_chimera_embedding(C, emb)
        dnx.draw_chimera_embedding(C, emb, embedded_graph=G)
        dnx.draw_chimera_embedding(C, emb, interaction_edges=C.edges())
