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
import networkx as nx
import dimod
import dwave.graphs as dnx


class TestIsClique(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()
        self.assertTrue(dnx.is_clique(G, []))

    def test_K1(self):
        G = nx.complete_graph(1)
        self.assertTrue(dnx.is_clique(G, [0]))
        self.assertTrue(dnx.is_clique(G, []))

    def test_K2(self):
        G = nx.complete_graph(2)
        self.assertTrue(dnx.is_clique(G, [0]))
        self.assertTrue(dnx.is_clique(G, []))
        self.assertTrue(dnx.is_clique(G, [0, 1]))

    def test_path3(self):
        G = nx.path_graph(3)
        self.assertTrue(dnx.is_clique(G, [0]))
        self.assertTrue(dnx.is_clique(G, [0, 1]))
        self.assertTrue(dnx.is_clique(G, []))
        self.assertFalse(dnx.is_clique(G, [0, 2]))


class TestMaxClique(unittest.TestCase):
    def test_maximum_independent_set_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
        G = dnx.chimera_graph(1, 2, 2)
        clique = dnx.maximum_clique(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_clique(G, clique))

        G = nx.path_graph(5)
        clique = dnx.maximum_clique(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_clique(G, clique))

    def test_default_sampler(self):
        G = nx.complete_graph(5)
        dnx.set_default_sampler(dimod.ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)

        clique = dnx.maximum_clique(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None,
                         "sampler did not unset correctly")

    def test_two_cliques(self):
        # This graph has two major cliques [0,1,2,3,4] and [11,12,13,14]
        # but the first one is bigger so that's the maximum_clique.
        G = nx.complete_graph(5)
        nx.add_path(G, [4, 5, 6, 7, 8])
        nx.add_path(G, [2, 9, 10])
        nx.add_path(G, [9, 11])
        nx.add_path(G, [11, 12, 13, 14, 11])
        nx.add_path(G, [12, 14])
        nx.add_path(G, [13, 11])
        clique = dnx.maximum_clique(G, dimod.ExactSolver())
        self.assertEqual(clique, [0, 1, 2, 3, 4])


class TestCliqueNumber(unittest.TestCase):
    def test_complete_graph(self):
        # In a complete graph the whole graph is a clique so every vertex is a
        # part of the clique.
        G = nx.complete_graph(17)
        clique_number = dnx.clique_number(G, dimod.ExactSolver())
        self.assertEqual(clique_number, 17)

    def test_two_cliques_num(self):
        # This graph has two major cliques [0,1,2,3,4] and [11,12,13,14]
        # but the first one is bigger so that's the maximum_clique.
        G = nx.complete_graph(5)
        nx.add_path(G, [4, 5, 6, 7, 8])
        nx.add_path(G, [2, 9, 10])
        nx.add_path(G, [9, 11])
        nx.add_path(G, [11, 12, 13, 14, 11])
        nx.add_path(G, [12, 14])
        nx.add_path(G, [13, 11])
        clique_number = dnx.clique_number(G, dimod.ExactSolver())
        self.assertEqual(clique_number, 5)
