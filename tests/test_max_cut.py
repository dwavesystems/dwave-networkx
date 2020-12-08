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
import dwave_networkx as dnx

from dimod import ExactSolver, SimulatedAnnealingSampler, qubo_energy


class TestMaxCut(unittest.TestCase):
    # def test_edge_cases(self):
    #     # get the empty graph
    #     G = nx.Graph()

    #     S = dnx.maximum_cut(G, ExactSolver())
    #     self.assertTrue(len(S) == 0)

    #     S = dnx.weighted_maximum_cut(G, ExactSolver())
    #     self.assertTrue(len(S) == 0)

    def test_typical_cases(self):

        G = nx.complete_graph(10)

        S = dnx.maximum_cut(G, ExactSolver())
        self.assertTrue(len(S) == 5)  # half of the nodes

        with self.assertRaises(dnx.DWaveNetworkXException):
            S = dnx.weighted_maximum_cut(G, ExactSolver())

        nx.set_edge_attributes(G, 1, 'weight')
        S = dnx.weighted_maximum_cut(G, ExactSolver())
        self.assertTrue(len(S) == 5)  # half of the nodes

        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (2, 4)])
        S = dnx.maximum_cut(G, ExactSolver())
        self.assertTrue(len(S) in (2, 3))

        # this needs another one for weight
