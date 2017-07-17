import unittest

from collections import defaultdict
from itertools import chain, combinations

import dwave_networkx as dnx

from dwave_networkx.algorithms.tests.solver import Sampler, sampler_found

from dwave_networkx.algorithms.matching_qa import _matching_qubo, _maximal_matching_qubo
from dwave_networkx.algorithms.matching_qa import is_matching, is_maximal_matching


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

            if is_matching(potential_matching):
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

            if is_matching(potential_matching):
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

            if is_maximal_matching(G, potential_matching):
                self.assertEqual(qubo_energy(Q, sample), ground_energy)
            elif not is_matching(potential_matching):
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

            if is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
            elif not is_matching(potential_matching):
                # for now we don't care about these, they should be covered by the _matching_qubo
                # part of the QUBO function
                pass
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B - infeasible_gap, 10**-8)

    def test_maximal_matching_combinated_qubo(self):
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

            if is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
                print sample, qubo_energy(Q, sample)
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

            if is_maximal_matching(G, potential_matching):
                # print potential_matching, qubo_energy(Q, sample)
                self.assertLess(abs(qubo_energy(Q, sample) - ground_energy), 10**-8)
            else:
                en = qubo_energy(Q, sample)

                gap = en - ground_energy
                if gap < infeasible_gap:
                    infeasible_gap = gap

        self.assertLessEqual(B - infeasible_gap, 10**-8)


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
