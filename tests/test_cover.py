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

import unittest
import random

import networkx as nx
import dwave.graphs as dnx

from dimod import ExactSolver, SimulatedAnnealingSampler


class TestCover(unittest.TestCase):

    def test_vertex_cover_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
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

    def test_vertex_cover_weighted(self):
        weight = 'weight'
        G = nx.path_graph(6)

        # favor even nodes
        nx.set_node_attributes(G, {node: node % 2 + 1 for node in G}, weight)
        cover = dnx.min_weighted_vertex_cover(G, weight, ExactSolver())
        self.assertEqual(set(cover), {0, 2, 4})

        # favor odd nodes
        nx.set_node_attributes(G, {node: (node + 1) % 2 + 1 for node in G}, weight)
        cover = dnx.min_weighted_vertex_cover(G, weight, ExactSolver())
        self.assertEqual(set(cover), {1, 3, 5})

        # make nodes 1 and 4 unlikely
        nx.set_node_attributes(G, {0: 1, 1: 3, 2: 1, 3: 1, 4: 3, 5: 1}, weight)
        cover = dnx.min_weighted_vertex_cover(G, weight, ExactSolver())
        self.assertEqual(set(cover), {0, 2, 3, 5})

        for __ in range(10):
            G = nx.gnp_random_graph(5, .5)
            nx.set_node_attributes(G, {node: random.random() for node in G}, weight)
            cover = dnx.min_weighted_vertex_cover(G, weight, ExactSolver())
            self.vertex_cover_check(G, cover)

    def test_default_sampler(self):
        G = nx.complete_graph(5)

        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        cover = dnx.min_vertex_cover(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    def test_dimod_vs_list(self):
        G = nx.path_graph(5)

        cover = dnx.min_vertex_cover(G, ExactSolver())
        cover = dnx.min_vertex_cover(G, SimulatedAnnealingSampler())

    def vertex_cover_check(self, G, cover):
        # each node in the vertex cover should be in G
        self.assertTrue(dnx.is_vertex_cover(G, cover))
