import unittest
import itertools

import networkx as nx
import dwave_networkx as dnx
from dwave_networkx.utils.test_samplers import ExactSolver, FastSampler


class TestSocial(unittest.TestCase):
    def test_network_imbalance_basic(self):
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

        frustrated_edges, colors = dnx.network_imbalance(S, sampler)
        self.check_bicolor(colors)

        greenteam = ['Ted']
        for p0 in set(S.nodes):
            for p1 in greenteam:
                S.add_edge(p0, p1, sign=1)

        frustrated_edges, colors = dnx.network_imbalance(S, sampler)
        self.check_bicolor(colors)

    def test_network_imbalance_docstring_example(self):
        sampler = ExactSolver()

        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
        frustrated_edges, colors = dnx.network_imbalance(S, sampler)
        self.check_bicolor(colors)
        self.assertEqual(frustrated_edges, {})
        S.add_edge('Ted', 'Bob', sign=1)  # Ted is friendly with all
        S.add_edge('Ted', 'Alice', sign=1)
        S.add_edge('Ted', 'Eve', sign=1)
        frustrated_edges, colors = dnx.network_imbalance(S, sampler)
        self.check_bicolor(colors)
        self.assertTrue(frustrated_edges == {('Ted', 'Eve'): {'sign': 1}} or
                        frustrated_edges == {('Eve', 'Ted'): {'sign': 1}})

    def check_bicolor(self, colors):
        # colors should be ints and either 0 or 1
        for c in colors.values():
            self.assertIsInstance(c, int)
            self.assertTrue(c in (0, 1))

    def test_default_sampler(self):
        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile

        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        frustrated_edges, colors = dnx.network_imbalance(S)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    @unittest.skipIf(FastSampler is None, "no dimod sampler provided")
    def test_dimod_vs_list(self):
        S = nx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile

        frustrated_edges, colors = dnx.network_imbalance(S, ExactSolver())
        frustrated_edges, colors = dnx.network_imbalance(S, FastSampler())
