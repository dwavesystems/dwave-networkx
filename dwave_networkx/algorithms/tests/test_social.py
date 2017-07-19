import unittest
import itertools

import dwave_networkx as dnx
from dwave_networkx.algorithms.tests.solver import Sampler, sampler_found


class TestSocial(unittest.TestCase):
    @unittest.skipIf(not sampler_found, "No solver found to test with")
    def test_network_imbalance_basic(self):

        blueteam = ['Alice', 'Bob', 'Carol']
        redteam0 = ['Eve']
        redteam1 = ['Mallory', 'Trudy']

        S = dnx.Graph()
        for p0, p1 in itertools.combinations(blueteam, 2):
            S.add_edge(p0, p1, sign=1)
        S.add_edge(*redteam1, sign=1)
        for p0 in blueteam:
            for p1 in redteam0:
                S.add_edge(p0, p1, sign=-1)
            for p1 in redteam1:
                S.add_edge(p0, p1, sign=-1)

        colors, frustrated_edges = dnx.network_imbalance_qubo(S, Sampler())

        greenteam = ['Ted']
        for p0 in S.nodes():
            for p1 in greenteam:
                S.add_edge(p0, p1, sign=1)

        colors, frustrated_edges = dnx.network_imbalance_qubo(S, Sampler())

    def test_network_imbalance_docstring_example(self):
        sampler = Sampler()

        S = dnx.Graph()
        S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
        S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
        S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
        frustrated_edges, colors = dnx.network_imbalance_qubo(S, sampler)
        self.assertEqual(frustrated_edges, {})
        S.add_edge('Ted', 'Bob', sign=1)  # Ted is friendly with all
        S.add_edge('Ted', 'Alice', sign=1)
        S.add_edge('Ted', 'Eve', sign=1)
        frustrated_edges, colors = dnx.network_imbalance_qubo(S, sampler)
        self.assertEqual(frustrated_edges, {('Ted', 'Eve'): {'sign': 1}})
