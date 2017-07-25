import unittest

import dwave_networkx as dnx

from dwave_networkx.algorithms.tests.solver import Sampler, sampler_found


class TestCover(unittest.TestCase):

    @unittest.skipIf(not sampler_found, "No solver found to test with")
    def test_vertex_cover_basic(self):

        G = dnx.chimera_graph(2, 2, 4)
        cover = dnx.min_vertex_cover_dm(G, Sampler())
        self.vertex_cover_check(G, cover)

        G = dnx.path_graph(5)
        cover = dnx.min_vertex_cover_dm(G, Sampler())
        self.vertex_cover_check(G, cover)

        for __ in range(10):
            G = dnx.gnp_random_graph(20, .5)
            cover = dnx.min_vertex_cover_dm(G, Sampler())
            self.vertex_cover_check(G, cover)

#######################################################################################
# Helper functions
#######################################################################################

    def vertex_cover_check(self, G, cover):
        # each node in the vertex cover should be in G
        self.assertTrue(all(node in G for node in cover))

        # a vertex cover should contain at least one of the nodes for each edge
        for (node1, node2) in G.edges():
            self.assertTrue(node1 in cover or node2 in cover)
