import unittest

import dwave_networkx as dnx

from dwave_networkx.algorithms.tests.solver import Solver


class TestColor(unittest.TestCase):
    def test_vertex_color_basic(self):

        G = dnx.chimera_graph(1, 2, 2)
        coloring = dnx.min_vertex_coloring_qa(G, Solver())
        self.vertex_coloring_check(G, coloring)

        G = dnx.path_graph(5)
        coloring = dnx.min_vertex_coloring_qa(G, Solver())
        self.vertex_coloring_check(G, coloring)

        for __ in range(10):
            G = dnx.gnp_random_graph(5, .5)
            coloring = dnx.min_vertex_coloring_qa(G, Solver())
            self.vertex_coloring_check(G, coloring)

    def test_vertex_color_complete_graph(self):
        G = dnx.complete_graph(101)
        coloring = dnx.min_vertex_coloring_qa(G, Solver())
        self.vertex_coloring_check(G, coloring)

    def test_vertex_color_odd_cycle_graph(self):
        """Graph that is an odd circle"""
        G = dnx.cycle_graph(11)
        coloring = dnx.min_vertex_coloring_qa(G, Solver())
        self.vertex_coloring_check(G, coloring)

    def test_vertex_color_no_edge_graph(self):
        """Graph with many nodes but no edges, should be caught before QUBO"""
        G = dnx.Graph()
        G.add_nodes_from(range(100))
        coloring = dnx.min_vertex_coloring_qa(G, Solver())
        self.vertex_coloring_check(G, coloring)

#######################################################################################
# Helper functions
#######################################################################################

    def vertex_coloring_check(self, G, coloring):
        for (node1, node2) in G.edges():
            self.assertNotEqual(coloring[node1], coloring[node2])