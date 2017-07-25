import unittest

from collections import defaultdict
from itertools import chain, combinations

import dwave_networkx as dnx

from dwave_networkx.algorithms_extended.tests.samplers import ExactSolver, FastSampler

from dwave_networkx.algorithms_extended.matching import _matching_qubo, _maximal_matching_qubo
from dwave_networkx.algorithms_extended.matching import _edge_mapping


class TestMatching(unittest.TestCase):

    def test__matching_qubo(self):
        # _matching_qubo creates a qubo that gives a matching for the given graph.
        # let's check that the solutions are all matchings

        G = dnx.complete_graph(5)
        MAG = .75  # magnitude arg for _matching_qubo

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Q = _matching_qubo(G, edge_mapping, magnitude=MAG)

        # now for each combination of ege, we check that if the combination
        # is a matching, it has qubo_energy 0, otherwise greater than 0. Which
        # is the desired behaviour
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_matching(potential_matching):
                self.assertEqual(qubo_energy(Q, sample), 0.)
            else:
                en = qubo_energy(Q, sample)
                if en < infeasible_gap:
                    infeasible_gap = en
                self.assertGreaterEqual(en, MAG)

        self.assertEqual(MAG, infeasible_gap)

        #
        # Another graph, Chimera tile this time
        #

        G = dnx.chimera_graph(1, 1, 4)
        MAG = .67

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Q = _matching_qubo(G, edge_mapping, magnitude=MAG)

        # now for each combination of ege, we check that if the combination
        # is a matching, it has qubo_energy 0, otherwise greater than 0. Which
        # is the desired behaviour
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_matching(potential_matching):
                self.assertEqual(qubo_energy(Q, sample), 0.)
            else:
                en = qubo_energy(Q, sample)
                if en < infeasible_gap:
                    infeasible_gap = en
                self.assertGreaterEqual(en, MAG)

        self.assertEqual(MAG, infeasible_gap)

    def test__maximal_matching_qubo(self):

        G = dnx.complete_graph(5)
        B = 1  # magnitude arg for _maximal_matching_qubo

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Q = _maximal_matching_qubo(G, edge_mapping, magnitude=B)

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, it has energy magnitude * |edges|
        ground_energy = -1. * B * len(G.edges())
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_maximal_matching(G, potential_matching):
                self.assertEqual(qubo_energy(Q, sample), ground_energy)
            elif not dnx.is_matching(potential_matching):
                # for now we don't care about these, they should be covered by the _matching_qubo
                # part of the QUBO function
                pass
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B, infeasible_gap)

        #
        # Another graph, Chimera tile this time
        #

        G = dnx.chimera_graph(1, 2, 2)
        B = 1  # magnitude arg for _maximal_matching_qubo

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Q = _maximal_matching_qubo(G, edge_mapping, magnitude=B)

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, it has energy magnitude * |edges|
        ground_energy = -1. * B * len(G.edges())
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
            elif not dnx.is_matching(potential_matching):
                # for now we don't care about these, they should be covered by the _matching_qubo
                # part of the QUBO function
                pass
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B - infeasible_gap, 10**-8)

    def test_maximal_matching_combined_qubo(self):
        """combine the qubo's generated by _maximal_matching_qubo and _matching_qubo
        and make sure they have the correct infeasible gap"""

        G = dnx.complete_graph(5)
        delta = max(G.degree(node) for node in G)  # maximum degree
        A = 1  # magnitude arg for _matching_qubo
        B = .75 * A / (delta - 2.)  # magnitude arg for _maximal_matching_qubo

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Qm = _matching_qubo(G, edge_mapping, magnitude=A)
        Qmm = _maximal_matching_qubo(G, edge_mapping, magnitude=B)

        Q = defaultdict(float)
        for edge, bias in Qm.items():
            Q[edge] += bias
        for edge, bias in Qmm.items():
            Q[edge] += bias
        Q = dict(Q)

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, and if so that is has ground energy, else
        # there is an infeasible gap
        ground_energy = -1. * B * len(G.edges())  # from maximal matching
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B - infeasible_gap, 10**-8)

        #
        # Another graph, Chimera tile this time
        #

        G = dnx.chimera_graph(1, 1, 4)
        delta = max(G.degree(node) for node in G)  # maximum degree
        A = 1  # magnitude arg for _matching_qubo
        B = .95 * A / (delta - 2.)  # magnitude arg for _maximal_matching_qubo

        edge_mapping = {edge: idx for idx, edge in enumerate(G.edges_iter())}
        edge_mapping.update({(e1, e0): idx for (e0, e1), idx in edge_mapping.items()})
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Qm = _matching_qubo(G, edge_mapping, magnitude=A)
        Qmm = _maximal_matching_qubo(G, edge_mapping, magnitude=B)

        Q = defaultdict(float)
        for edge, bias in Qm.items():
            Q[edge] += bias
        for edge, bias in Qmm.items():
            Q[edge] += bias
        Q = dict(Q)

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, and if so that is has ground energy, else
        # there is an infeasible gap
        ground_energy = -1. * B * len(G.edges())  # from maximal matching
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            if dnx.is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B - infeasible_gap, 10**-8)

    def test_maximal_matching_combined_qubo_bug1(self):

        # a graph that was not working
        G = dnx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
        G.add_edges_from([(0, 1), (0, 3), (1, 3), (1, 4), (1, 6), (1, 7), (2, 3), (2, 5), (2, 7),
                          (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), (5, 7)])
        delta = max(G.degree(node) for node in G)  # maximum degree
        A = 1  # magnitude arg for _matching_qubo
        B = .95 * A / (delta - 2.)  # magnitude arg for _maximal_matching_qubo

        edge_mapping = _edge_mapping(G)
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Qmm = _maximal_matching_qubo(G, edge_mapping, magnitude=B)  # Q is a defaultdict
        Qm = _matching_qubo(G, edge_mapping, magnitude=A)
        Q = Qmm.copy()
        for edge, bias in Qm.items():
            Q[edge] += bias
        Q = dict(Q)  # we are not necessarily sure that the given sampler can handle a defaultdict
        Qmm = dict(Qmm)

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, and if so that is has ground energy, else
        # there is an infeasible gap
        ground_energy = -1. * B * len(G.edges())  # from maximal matching
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            en_matching = qubo_energy(Qm, sample)
            en_maximal = qubo_energy(Qmm, sample)
            en = qubo_energy(Q, sample)

            self.assertLess(abs(en_matching + en_maximal - en), 10**-8)

            if dnx.is_maximal_matching(G, potential_matching):
                # if the sample is a maximal matching, then let's check each qubo
                # and combined together
                self.assertEqual(en_matching, 0.0)  # matching
                self.assertLess(abs(en_maximal - ground_energy), 10**-8)

            elif dnx.is_matching(potential_matching):
                # in this case we expect the energy contribution of Qm to be 0
                self.assertEqual(en_matching, 0.0)  # matching

                # but there should be a gap to Qmm, because the matching is not maximal
                self.assertGreater(en_maximal, ground_energy)

                gap = en_matching + en_maximal - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap
            else:
                # ok, these are not even matching
                self.assertGreater(en_matching, 0)  # so matching energy should be > 0

                self.assertGreater(gap, 0)

                gap = en_matching + en_maximal - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertGreater(infeasible_gap, 0)

    def test_maximal_matching_combined_qubo_bug2(self):

        # a graph that was not working
        G = dnx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
        G.add_edges_from([(0, 3), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 7), (2, 4), (2, 6),
                          (3, 6), (3, 7), (4, 7), (6, 7)])
        delta = max(G.degree(node) for node in G)  # maximum degree
        A = 1  # magnitude arg for _matching_qubo
        B = .75 * A / (delta - 2.)  # magnitude arg for _maximal_matching_qubo

        edge_mapping = _edge_mapping(G)
        inv_edge_mapping = {idx: edge for edge, idx in edge_mapping.items()}

        Qmm = _maximal_matching_qubo(G, edge_mapping, magnitude=B)
        Qm = _matching_qubo(G, edge_mapping, magnitude=A)
        Q = Qmm.copy()
        for edge, bias in Qm.items():
            Q[edge] += bias

        # now for each combination of edges, we check that if the combination
        # is a maximal matching, and if so that is has ground energy, else
        # there is an infeasible gap
        ground_energy = -1. * B * len(G.edges())  # from maximal matching
        infeasible_gap = float('inf')
        for edge_vars in powerset(set(edge_mapping.values())):

            # get the matching from the variables
            potential_matching = {inv_edge_mapping[v] for v in edge_vars}

            # get the sample from the edge_vars
            sample = {v: 0 for v in edge_mapping.values()}
            for v in edge_vars:
                sample[v] = 1

            en_matching = qubo_energy(Qm, sample)
            en_maximal = qubo_energy(Qmm, sample)
            en = qubo_energy(Q, sample)

            self.assertLess(abs(en_matching + en_maximal - en), 10**-8)

            if dnx.is_maximal_matching(G, potential_matching):
                # if the sample is a maximal matching, then let's check each qubo
                # and combined together
                self.assertEqual(en_matching, 0.0)  # matching
                self.assertLess(abs(en_maximal - ground_energy), 10**-8)

            elif dnx.is_matching(potential_matching):
                # in this case we expect the energy contribution of Qm to be 0
                self.assertEqual(en_matching, 0.0)  # matching

                # but there should be a gap to Qmm, because the matching is not maximal
                self.assertGreater(en_maximal, ground_energy)

                gap = en_matching + en_maximal - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap
            else:
                # ok, these are not even matching
                self.assertGreater(en_matching, 0)  # so matching energy should be > 0

                self.assertGreater(gap, 0)

                gap = en_matching + en_maximal - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertGreater(infeasible_gap, 0)

    def test_maximal_matching_typical(self):

        G = dnx.complete_graph(5)
        matching = dnx.maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))

        for __ in range(10):
            G = dnx.gnp_random_graph(7, .5)
            matching = dnx.maximal_matching_dm(G, ExactSolver())
            self.assertTrue(dnx.is_maximal_matching(G, matching))

    def test_minimal_maximal_matching_typical(self):

        G = dnx.complete_graph(5)
        matching = dnx.minimal_maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))

        for __ in range(10):
            G = dnx.gnp_random_graph(7, .5)
            matching = dnx.minimal_maximal_matching_dm(G, ExactSolver())
            self.assertTrue(dnx.is_maximal_matching(G, matching))

    def test_path_graph(self):
        G = dnx.path_graph(10)
        matching = dnx.maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))

        matching = dnx.minimal_maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))

        G.add_edge(0, 9)

        matching = dnx.maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))

        matching = dnx.minimal_maximal_matching_dm(G, ExactSolver())
        self.assertTrue(dnx.is_maximal_matching(G, matching))


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def qubo_energy(Q, sample):
    """Calculate the quadratic polynomial value of the given sample
    to a quadratic unconstrained binary optimization (QUBO) problem.
    """
    energy = 0

    for v0, v1 in Q:
        energy += sample[v0] * sample[v1] * Q[(v0, v1)]

    return energy
