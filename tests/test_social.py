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
import itertools

import networkx as nx
import dwave.graphs as dnx
from dimod import ExactSolver, SimulatedAnnealingSampler


class TestSocial(unittest.TestCase):
    def check_bicolor(self, colors):
        # colors should be ints and either 0 or 1
        for c in colors.values():
            self.assertTrue(c in (0, 1))

    def test_structural_imbalance_basic(self):
        sampler = ExactSolver()

        blueteam = ['Alice', 'Bob', 'Carol']
        redteam0 = ['Eve']
        redteam1 = ['Mallory', 'Trudy']

        S = nx.Graph()
        for p0, p1 in itertools.combinations(blueteam, 2):
            S.add_edge(p0, p1, sign=1)
        S.add_edge(*redteam1, sign=1)
        for p0 in blueteam:
            for p1 in redteam0:
                S.add_edge(p0, p1, sign=-1)
            for p1 in redteam1:
                S.add_edge(p0, p1, sign=-1)

        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
        self.check_bicolor(colors)

        greenteam = ['Ted']
        for p0 in set(S.nodes):
            for p1 in greenteam:
                S.add_edge(p0, p1, sign=1)

        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
        self.check_bicolor(colors)

    def test_structural_imbalance_docstring_example(self):
        sampler = ExactSolver()

        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
        self.check_bicolor(colors)
        self.assertEqual(frustrated_edges, {})
        S.add_edge('Ted', 'Bob', sign=1)  # Ted is friendly with all
        S.add_edge('Ted', 'Alice', sign=1)
        S.add_edge('Ted', 'Eve', sign=1)
        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
        self.check_bicolor(colors)
        self.assertTrue(frustrated_edges == {('Ted', 'Eve'): {'sign': 1}} or
                        frustrated_edges == {('Eve', 'Ted'): {'sign': 1}})

    def test_default_sampler(self):
        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile

        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        frustrated_edges, colors = dnx.structural_imbalance(S)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    def test_invalid_graph(self):
        """should throw an error with a graph without sign attribute"""
        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)
        S.add_edge('Alice', 'Eve', sign=-1)
        S.add_edge('Bob', 'Eve')  # invalid edge

        with self.assertRaises(ValueError):
            frustrated_edges, colors = dnx.structural_imbalance(S, ExactSolver())

    def test_sign_zero(self):
        """though not documentented, agents with no relation can have sign 0.
        This is less performant than just not having an edge in most cases."""
        sampler = ExactSolver()

        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=0)  # Bob and Eve have no relation

        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)

        self.assertEqual(frustrated_edges, {})  # should be no frustration

    def test_frustrated_hostile_edge(self):
        """Set up a graph where the frustrated edge should be hostile"""
        sampler = ExactSolver()

        S = nx.florentine_families_graph()

        # set all hostile
        nx.set_edge_attributes(S, -1, 'sign')

        # smoke test
        frustrated_edges, colors = dnx.structural_imbalance(S, sampler)
        self.check_bicolor(colors)
