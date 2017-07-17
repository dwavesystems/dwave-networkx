import unittest
from collections import defaultdict

import dwave_networkx as dnx

from dwave_networkx.algorithms.tests.solver import Sampler, sampler_found

from dwave_networkx.algorithms.matching_qa import _matching_qubo, _maximal_matching_qubo


class TestMatching(unittest.TestCase):

    # @unittest.skipIf(not sampler_found, "No solver found to test with")
    # def test_maximal_matching_basic(self):

    #     G = dnx.chimera_graph(1, 2, 2)
    #     matching = dnx.minimal_maximal_matching(G, Sampler())
    #     self.check_matching(G, matching)

    #     G = dnx.path_graph(5)
    #     matching = dnx.minimal_maximal_matching(G, Sampler())
    #     self.check_matching(G, matching)

    #     for __ in range(10):
    #         G = dnx.gnp_random_graph(7, .3)
    #         matching = dnx.minimal_maximal_matching(G, Sampler())
    #         self.check_matching(G, matching)

    # def test_maximal_matching_bug1(self):
    #     G = dnx.Graph()
    #     G.add_nodes_from(range(7))
    #     G.add_edges_from([(1, 2), (1, 4), (2, 3), (2, 4), (2, 5), (4, 6)])
    #     matching = dnx.minimal_maximal_matching(G, Sampler())
    #     self.check_matching(G, matching)

    # def test_maximal_matching_qubo_helper(self):

    #     G = dnx.complete_graph(4)

    #     edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
    #     edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})

    #     delta = max(G.degree(node) for node in G)
    #     A = 1.
    #     B = .75 * A / (delta - 2)

    #     print B

    #     Q_m = _matching_qubo(G, edge_mapping, A)
    #     Q_mm = _maximal_matching_qubo(G, edge_mapping, B)

    #     Q = defaultdict(float)
    #     Q.update(Q_m)
    #     for key, bias in Q_mm.items():
    #         Q[key] += bias
    #     Q = dict(Q)

    #     # for each sample in the response that is at ground
    #     ground_energy = None
    #     for sample, en in Sampler().sample_qubo(Q).items():
    #         if ground_energy is None:
    #             ground_energy = en
    #         else:
    #             if en > ground_energy:
    #                 break

    #         matching = set(edge for edge in G.edges_iter() if sample[edge_mapping[edge]] > 0)
    #         self.check_matching(G, matching)
    #         self.check_maximal_matching(G, matching)

    def test_helper_matching_qubo_basic(self):
        # _matching_qubo creates a qubo that gives a matching for the given graph.
        # let's check that the solutions are all matchings

        G = dnx.complete_graph(6)

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})

        Q = _matching_qubo(G, edge_mapping)

        # for each sample in the response that is at ground
        ground_energy = None
        for sample, en in Sampler().sample_qubo(Q).items():
            if ground_energy is None:
                ground_energy = en
            else:
                if en > ground_energy:
                    break

            matching = set(edge for edge in G.edges_iter() if sample[edge_mapping[edge]] > 0)
            self.check_matching(G, matching)

    def check_matching(self, G, matching):
        """Confirm that the given matching is indeed a matching."""
        for e0 in matching:
            for e1 in matching:
                if e0 == e1:
                    continue
                self.assertFalse({n for n in e0} & {n for n in e1},
                                 '{} is not a matching'.format(matching))

    def check_maximal_matching(self, G, matching):
        """Confirm that the given matching is maximal. That is we cannot add any edges"""

        for (u, v) in G.edges_iter():
            if (u, v) in matching:
                continue
            if (v, u) in matching:
                continue

            self.assertTrue(all((v in edge) or (u in edge) for edge in matching),
                            '{} is not a maximal matching because we can add edge {}'.format(matching, (u, v)))
