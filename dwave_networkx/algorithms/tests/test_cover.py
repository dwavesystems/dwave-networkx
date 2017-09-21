import unittest

import networkx as nx
import dwave_networkx as dnx

from dwave_networkx.utils.test_samplers import ExactSolver, FastSampler


class TestCover(unittest.TestCase):

    def test_vertex_cover_basic(self):

        G = dnx.chimera_graph(1, 2, 2)
        cover = dnx.min_vertex_cover(G, ExactSolver())
        self.vertex_cover_check(G, cover)

        G = nx.path_graph(5)
        cover = dnx.min_vertex_cover(G, ExactSolver())
        self.vertex_cover_check(G, cover)

        for __ in range(10):
            G = nx.gnp_random_graph(5, .5)
            cover = dnx.min_vertex_cover(G, ExactSolver())
            self.vertex_cover_check(G, cover)

    def test_default_sampler(self):
        G = nx.complete_graph(5)

        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        cover = dnx.min_vertex_cover(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    @unittest.skipIf(FastSampler is None, "no dimod sampler provided")
    def test_dimod_vs_list(self):
        G = nx.path_graph(5)

        cover = dnx.min_vertex_cover(G, ExactSolver())
        cover = dnx.min_vertex_cover(G, FastSampler())

#######################################################################################
# Helper functions
#######################################################################################

    def vertex_cover_check(self, G, cover):
        # each node in the vertex cover should be in G
        self.assertTrue(dnx.is_vertex_cover(G, cover))
