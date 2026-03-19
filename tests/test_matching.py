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

import dimod
import networkx as nx
import parameterized

import dwave.graphs as dnx


@parameterized.parameterized_class(
    'graph',
    [[nx.Graph()],
     [nx.path_graph(10)],
     [nx.complete_graph(4)],
     [nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5)])],
     [nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (1, 5)])],
     [nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])],
     [nx.Graph({0: [], 1: [6], 2: [5], 3: [4], 4: [3], 2: [5], 6: [1]})]
     # [nx.Graph([(0, 3), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 7),
     #            (2, 4), (2, 6), (3, 6), (3, 7), (4, 7), (6, 7)])],  # slow
     ]
    )
class TestMatching(unittest.TestCase):
    def test_matching_bqm(self):
        bqm = dnx.matching_bqm(self.graph)

        # the ground states should be exactly the matchings of G
        sampleset = dimod.ExactSolver().sample(bqm)

        for sample, energy in sampleset.data(['sample', 'energy']):
            edges = [v for v, val in sample.items() if val > 0]
            self.assertEqual(nx.is_matching(self.graph, edges), energy == 0)
            self.assertTrue(energy == 0 or energy >= 1)

            # while we're at it, test deprecated is_matching
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(nx.is_matching(self.graph, edges),
                                 dnx.is_matching(edges))

    def test_maximal_matching(self):
        matching = dnx.algorithms.matching.maximal_matching(
            self.graph, dimod.ExactSolver())
        self.assertTrue(nx.is_maximal_matching(self.graph, matching))

    def test_maximal_matching_bqm(self):
        bqm = dnx.maximal_matching_bqm(self.graph)

        # the ground states should be exactly the maximal matchings of G
        sampleset = dimod.ExactSolver().sample(bqm)

        for sample, energy in sampleset.data(['sample', 'energy']):
            edges = set(v for v, val in sample.items() if val > 0)
            self.assertEqual(nx.is_maximal_matching(self.graph, edges),
                             energy == 0)
            self.assertGreaterEqual(energy, 0)

            # while we're at it, test deprecated is_maximal_matching
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(nx.is_maximal_matching(self.graph, edges),
                                 dnx.is_maximal_matching(self.graph, edges))

    def test_min_maximal_matching(self):
        matching = dnx.min_maximal_matching(self.graph, dimod.ExactSolver())
        self.assertTrue(nx.is_maximal_matching(self.graph, matching))

    def test_min_maximal_matching_bqm(self):
        bqm = dnx.min_maximal_matching_bqm(self.graph)

        if len(self.graph) == 0:
            self.assertEqual(len(bqm.linear), 0)
            return

        # the ground states should be exactly the minimum maximal matchings of
        # G
        sampleset = dimod.ExactSolver().sample(bqm)

        # we'd like to use sampleset.lowest() but it didn't exist in dimod
        # 0.8.0
        ground_energy = sampleset.first.energy
        cardinalities = set()
        for sample, energy in sampleset.data(['sample', 'energy']):
            if energy > ground_energy:
                continue
            edges = set(v for v, val in sample.items() if val > 0)
            self.assertTrue(nx.is_maximal_matching(self.graph, edges))
            cardinalities.add(len(edges))

        # all ground have the same cardinality (or it's empty)
        self.assertEqual(len(cardinalities), 1)
        cardinality, = cardinalities

        # everything that's not ground has a higher energy
        for sample, energy in sampleset.data(['sample', 'energy']):
            edges = set(v for v, val in sample.items() if val > 0)
            if energy != sampleset.first.energy:
                if nx.is_maximal_matching(self.graph, edges):
                    self.assertGreater(len(edges), cardinality)


class TestMinMaximalMatching(unittest.TestCase):
    def test_default_sampler(self):
        G = nx.complete_graph(5)

        dnx.set_default_sampler(dimod.ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        matching = dnx.algorithms.matching.maximal_matching(G)
        matching = dnx.min_maximal_matching(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None,
                         "sampler did not unset correctly")
