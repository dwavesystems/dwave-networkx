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


class TestIsIndependentSet(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()

        self.assertTrue(dnx.is_independent_set(G, []))

    def test_K1(self):
        G = nx.complete_graph(1)

        self.assertTrue(dnx.is_independent_set(G, [0]))
        self.assertTrue(dnx.is_independent_set(G, []))

    def test_K2(self):
        G = nx.complete_graph(2)

        self.assertTrue(dnx.is_independent_set(G, [0]))
        self.assertTrue(dnx.is_independent_set(G, []))
        self.assertFalse(dnx.is_independent_set(G, [0, 1]))

    def test_path3(self):
        G = nx.path_graph(3)

        self.assertTrue(dnx.is_independent_set(G, [0]))
        self.assertTrue(dnx.is_independent_set(G, [0, 2]))
        self.assertTrue(dnx.is_independent_set(G, []))
        self.assertFalse(dnx.is_independent_set(G, [0, 1]))


class TestWeightedMaximumIndependentSet(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()

        Q = dnx.maximum_weighted_independent_set_qubo(G)

        self.assertEqual(Q, {})

    def test_K1_no_weights(self):
        G = nx.complete_graph(1)

        Q = dnx.maximum_weighted_independent_set_qubo(G)

        self.assertEqual(Q, {(0, 0): -1})

    def test_K1_weighted(self):
        G = nx.Graph()
        G.add_node(0, weight=.5)

        Q = dnx.maximum_weighted_independent_set_qubo(G)

        self.assertEqual(Q, {(0, 0): -1.})  # should be scaled to 1

    def test_K2_weighted(self):
        G = nx.Graph()
        G.add_node(0, weight=.5)
        G.add_node(1, weight=1)
        G.add_edge(0, 1)

        Q = dnx.maximum_weighted_independent_set_qubo(G, weight='weight')

        self.assertEqual(Q, {(0, 0): -.5, (1, 1): -1, (0, 1): 2.0})

    def test_K2_partially_weighted(self):
        G = nx.Graph()
        G.add_node(0, weight=.5)
        G.add_node(1)
        G.add_edge(0, 1)

        Q = dnx.maximum_weighted_independent_set_qubo(G, weight='weight')

        self.assertEqual(Q, {(0, 0): -.5, (1, 1): -1, (0, 1): 2.0})

    def test_path3_weighted(self):
        G = nx.path_graph(3)
        G.nodes[1]['weight'] = 2.1

        Q = dnx.maximum_weighted_independent_set_qubo(G, weight='weight')

        self.assertLess(dimod.qubo_energy({0: 0, 1: 1, 2: 0}, Q),
                        dimod.qubo_energy({0: 1, 1: 0, 2: 1}, Q))


class TestIndepSet(unittest.TestCase):

    def test_maximum_independent_set_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
        G = dnx.chimera_graph(1, 2, 2)
        indep_set = dnx.maximum_independent_set(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_independent_set(G, indep_set))

        G = nx.path_graph(5)
        indep_set = dnx.maximum_independent_set(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_independent_set(G, indep_set))

    def test_maximum_independent_set_weighted(self):
        weight = 'weight'
        G = nx.path_graph(6)

        # favor odd nodes
        nx.set_node_attributes(G, {node: node % 2 + 1 for node in G}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, dimod.ExactSolver())
        self.assertEqual(set(indep_set), {1, 3, 5})

        # favor even nodes
        nx.set_node_attributes(G, {node: (node + 1) % 2 + 1 for node in G}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, dimod.ExactSolver())
        self.assertEqual(set(indep_set), {0, 2, 4})

        # make nodes 1 and 4 likely
        nx.set_node_attributes(G, {0: 1, 1: 3, 2: 1, 3: 1, 4: 3, 5: 1}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, dimod.ExactSolver())
        self.assertEqual(set(indep_set), {1, 4})

    def test_default_sampler(self):
        G = nx.complete_graph(5)

        dnx.set_default_sampler(dimod.ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        indep_set = dnx.maximum_independent_set(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    def test_dimod_vs_list(self):
        G = nx.path_graph(5)

        indep_set = dnx.maximum_independent_set(G, dimod.ExactSolver())
        indep_set = dnx.maximum_independent_set(G, dimod.SimulatedAnnealingSampler())
