import unittest
import random

# import os
# import matplotlib.pyplot as plt

import networkx as nx
import dwave_networkx as dnx
from dwave_networkx.algorithms.independent_set import _weighted_independent_sets

from dimod import ExactSolver, SimulatedAnnealingSampler


#######################################################################################
# Unit Tests
#######################################################################################

class TestIndepSet(unittest.TestCase):

    def test_maximum_independent_set_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
        G = dnx.chimera_graph(1, 2, 2)
        indep_set = dnx.maximum_independent_set(G, ExactSolver())
        self.set_independence_check(G, indep_set)

        G = nx.path_graph(5)
        indep_set = dnx.maximum_independent_set(G, ExactSolver())
        self.set_independence_check(G, indep_set)

        for __ in range(10):
            G = nx.gnp_random_graph(5, .5)
            indep_set = dnx.maximum_independent_set(G, ExactSolver())
            self.set_independence_check(G, indep_set)

    def test_maximum_independent_set_weighted(self):
        weight = 'weight'
        G = nx.path_graph(6)

        # favor odd nodes
        nx.set_node_attributes(G, {node: node % 2 + 1 for node in G}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, ExactSolver())
        self.assertEqual(set(indep_set), {1, 3, 5})

        # favor even nodes
        nx.set_node_attributes(G, {node: (node + 1) % 2 + 1 for node in G}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, ExactSolver())
        self.assertEqual(set(indep_set), {0, 2, 4})

        # make nodes 1 and 4 likely
        nx.set_node_attributes(G, {0: 1, 1: 3, 2: 1, 3: 1, 4: 3, 5: 1}, weight)
        indep_set = dnx.maximum_weighted_independent_set(G, weight, ExactSolver())
        self.assertEqual(set(indep_set), {1, 4})

        for __ in range(10):
            G = nx.gnp_random_graph(5, .5)
            nx.set_node_attributes(G, {node: random.random() for node in G}, weight)
            indep_set = dnx.maximum_weighted_independent_set(G, weight, ExactSolver())
            self.set_independence_check(G, indep_set)

    def test_default_sampler(self):
        G = nx.complete_graph(5)

        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        indep_set = dnx.maximum_independent_set(G)
        dnx.unset_default_sampler()
        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    def test_dimod_vs_list(self):
        G = nx.path_graph(5)

        indep_set = dnx.maximum_independent_set(G, ExactSolver())
        indep_set = dnx.maximum_independent_set(G, SimulatedAnnealingSampler())

    def test__weighted_independent_sets(self):
        weight = 'weight'

        for i in range(10):
            G = nx.gnp_random_graph(5, .5)
            nx.set_node_attributes(G, {node: random.random() for node in G}, weight)
            response = _weighted_independent_sets(G, weight, ExactSolver())

            # Independent sets should be ordered from highest to lowest total weight
            prevSum = float('Inf')
            for sample in response.samples():
                candidate_set = [node for node in sample if sample[node] > 0]
                if dnx.is_independent_set(G, candidate_set):
                    curSum = sum((attributes[weight] for v, attributes in G.nodes(data=True) if v in candidate_set))
                    self.assertGreaterEqual(prevSum, curSum)
                    prevSum = curSum

            # # Draw graphs
            # pos = nx.spring_layout(G)
            # labels = nx.get_node_attributes(G, weight)
            # weights = labels.values()
            # minweight = min(weights)
            # node_size = [x / minweight * 300 for x in weights]
            # for j, (sample, energy) in enumerate(response.items()):
            #     directory_path = os.path.join('tmp', '%d' % i)
            #     if not os.path.exists(directory_path):
            #         os.makedirs(directory_path)
            #     file_path = os.path.join(directory_path, '%s.png' % j)
            #     nx.draw_networkx(G, pos, node_size=node_size, node_color=list(
            #         sample.values()), cmap="tab10", labels=sample)
            #     plt.title(energy)
            #     plt.savefig(file_path)
            #     plt.clf()


#######################################################################################
# Helper functions
#######################################################################################

    def set_independence_check(self, G, indep_set):
        """Check that the given set of nodes are in fact nodes in the graph and
        independent of eachother (that is there are no edges between them"""
        self.assertTrue(dnx.is_independent_set(G, indep_set))
