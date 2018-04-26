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
    def test_pegasus_layout_coords(self):
        G = dnx.pegasus_graph(2, coordinates=True)
        pos = dnx.pegasus_layout(G)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_pegasus_layout_ints(self):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_pegasus_layout_ints_nodata(self, data=False):
        G = dnx.pegasus_graph(2)
        pos = dnx.pegasus_layout(G)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_draw_pegasus(self, data=False):
        G = dnx.pegasus_graph(2)
        h = {v: v%12 for v in G}
        J = {(u,v): (u+v)%24 for u, v in G.edges()}

        dnx.draw_pegasus(G, linear_biases=h, quadratic_biases=J)

    @unittest.skipIf(not _numpy, "No numpy")
    def test_draw_pegasus_embedding(self, data=False):
        P = dnx.pegasus_graph(2)
        G = nx.grid_graph([3,3,2])
        emb = {(0, 0, 0): [35], (0, 0, 1): [12], (0, 0, 2): [31], (0, 1, 0): [16],
               (0, 1, 1): [36], (0, 1, 2): [11], (0, 2, 0): [39], (0, 2, 1): [6],
               (0, 2, 2): [41], (1, 0, 0): [34], (1, 0, 1): [13], (1, 0, 2): [30], 
               (1, 1, 0): [17], (1, 1, 1): [37], (1, 1, 2): [10], (1, 2, 0): [38],
               (1, 2, 1): [7], (1, 2, 2): [40]}
        dnx.draw_pegasus_embedding(P, emb)
        dnx.draw_pegasus_embedding(P, emb, embedded_graph=G)
        dnx.draw_pegasus_embedding(P, emb, interaction_edges=G.edges())

