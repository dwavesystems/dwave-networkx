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
from dimod import ExactDQMSolver


class TestPartitioning(unittest.TestCase):
    # def test_edge_cases(self):
    #     # get the empty graph
    #     G = nx.Graph()

    #     node_partitions = dnx.partition(G, sampler=ExactDQMSolver())
    #     self.assertTrue(node_partitions == {})

    #     node_partitions = dnx.weighted_partition(G, sampler=ExactDQMSolver())
    #     self.assertTrue(node_partitions == {})

    def test_typical_cases(self):

        G = nx.complete_graph(10)

        node_partitions = dnx.partition(G, num_partitions=5, sampler=ExactDQMSolver())
        for i in range(5):
            self.assertTrue(sum(x == i for x in node_partitions.values()) == 2) # 5 equally sized subsets

        node_partitions = dnx.partition(G, sampler=ExactDQMSolver())
        self.assertTrue(sum(x == 0 for x in node_partitions.values()) == 5)  # half of the nodes in subset '0'

        with self.assertRaises(dnx.DWaveNetworkXException):
            node_partitions = dnx.weighted_maximum_cut(G, sampler=ExactDQMSolver())

        nx.set_edge_attributes(G, 1, 'weight')
        node_partitions = dnx.weighted_maximum_cut(G, sampler=ExactDQMSolver())
        self.assertTrue(sum(x == 0 for x in node_partitions.values()) == 5)  # half of the nodes in subset '0'

        node_partitions = dnx.weighted_partition(G, num_partitions=5, sampler=ExactDQMSolver())
        for i in range(5):
            self.assertTrue(sum(x == i for x in node_partitions.values()) == 2) # 5 equally sized subsets

        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (3, 4), (2, 4)])
        node_partitions = dnx.partition(G, sampler=ExactDQMSolver())
        self.assertTrue(sum(x == 0 for x in node_partitions.values()) in (2, 3)) # either 2 or 3 nodes in subset '0'