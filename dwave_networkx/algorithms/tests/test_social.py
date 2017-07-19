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
        print colors, frustrated_edges
