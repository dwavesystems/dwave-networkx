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

import itertools
import unittest

import networkx as nx

import dimod

import dwave.graphs as dnx


# adapted from future networkx version (nx.path_weight(...))
def path_weight(G, path):
    '''Return the total cost of a cycle in G specified by path.'''
    cost = 0
    for node, nbr in nx.utils.pairwise(path):
        cost += G[node][nbr]['weight']
    # add back to the starting point
    cost += G[path[-1]][path[0]]['weight']
    return cost


class TestIsHamiltonCycle(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()

        self.assertTrue(dnx.is_hamiltonian_path(G, []))

    def test_K1(self):
        G = nx.complete_graph(1)

        self.assertTrue(dnx.is_hamiltonian_path(G, [0]))
        self.assertFalse(dnx.is_hamiltonian_path(G, []))

    def test_K2(self):
        G = nx.complete_graph(2)

        self.assertTrue(dnx.is_hamiltonian_path(G, [0, 1]))
        self.assertTrue(dnx.is_hamiltonian_path(G, [1, 0]))
        self.assertFalse(dnx.is_hamiltonian_path(G, [0]))
        self.assertFalse(dnx.is_hamiltonian_path(G, [1]))
        self.assertFalse(dnx.is_hamiltonian_path(G, []))

    def test_K3(self):
        G = nx.complete_graph(3)

        self.assertTrue(dnx.is_hamiltonian_path(G, [0, 1, 2]))
        self.assertTrue(dnx.is_hamiltonian_path(G, [1, 0, 2]))
        self.assertFalse(dnx.is_hamiltonian_path(G, [0, 1]))
        self.assertFalse(dnx.is_hamiltonian_path(G, [0]))
        self.assertFalse(dnx.is_hamiltonian_path(G, [1]))
        self.assertFalse(dnx.is_hamiltonian_path(G, []))


class TestTSP(unittest.TestCase):

    def test_TSP_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        route = dnx.traveling_salesperson(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_hamiltonian_path(G, route))

        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = u+v
        route = dnx.traveling_salesperson(G, dimod.ExactSolver(), lagrange=10.0)
        self.assertTrue(dnx.is_hamiltonian_path(G, route))

    def test_dimod_vs_list(self):
        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1

        route = dnx.traveling_salesperson(G, dimod.ExactSolver())
        route = dnx.traveling_salesperson(G, dimod.SimulatedAnnealingSampler())

    def test_weighted_complete_graph(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 3),
                                   (1, 3, 4), (2, 3, 5)})
        route = dnx.traveling_salesperson(G, dimod.ExactSolver(), lagrange=10)

        self.assertEqual(len(route), len(G))

    def test_start(self):
        G = nx.Graph()
        G.add_weighted_edges_from((u, v, .5)
                                  for u, v in itertools.combinations(range(3), 2))

        route = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=1)

        self.assertEqual(route[0], 1)

    def test_weighted_complete_digraph(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([
            (0, 1, 2),
            (1, 0, 1),
            (0, 2, 2),
            (2, 0, 2),
            (0, 3, 1),
            (3, 0, 2),
            (1, 2, 2),
            (2, 1, 1),
            (1, 3, 2),
            (3, 1, 2),
            (2, 3, 2),
            (3, 2, 1),
        ])

        route = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=1)

        self.assertEqual(len(route), len(G))
        self.assertListEqual(route, [1, 0, 3, 2])

        cost = path_weight(G, route)

        self.assertEqual(cost, 4)


class TestTSPQUBO(unittest.TestCase):
    def test_empty(self):
        Q = dnx.traveling_salesperson_qubo(nx.Graph())
        self.assertEqual(Q, {})

    def test_k3(self):
        # 3cycle so all paths are equally good
        G = nx.Graph()
        G.add_weighted_edges_from([('a', 'b', 0.5),
                                   ('b', 'c', 1.0),
                                   ('a', 'c', 2.0)])

        Q = dnx.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # all routes are min weight
        min_routes = list(itertools.permutations(G.nodes))

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm.variables}
            for idx, city in enumerate(route):
                sample[(city, idx)] = 1
            self.assertAlmostEqual(bqm.energy(sample), ground_energy)

        # all min-energy solutions are valid routes
        ground_count = 0
        for sample, energy in sampleset.data(['sample', 'energy']):
            if abs(energy - ground_energy) > .001:
                break
            ground_count += 1

        self.assertEqual(ground_count, len(min_routes))

    def test_k3_bidirectional(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([('a', 'b', 0.5),
                                   ('b', 'a', 0.5),
                                   ('b', 'c', 1.0),
                                   ('c', 'b', 1.0),
                                   ('a', 'c', 2.0),
                                   ('c', 'a', 2.0)])

        Q = dnx.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # all routes are min weight
        min_routes = list(itertools.permutations(G.nodes))

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm.variables}
            for idx, city in enumerate(route):
                sample[(city, idx)] = 1
            self.assertAlmostEqual(bqm.energy(sample), ground_energy)

        # all min-energy solutions are valid routes
        ground_count = 0
        for sample, energy in sampleset.data(['sample', 'energy']):
            if abs(energy - ground_energy) > .001:
                break
            ground_count += 1

        self.assertEqual(ground_count, len(min_routes))

    def test_graph_missing_edges(self):
        G1 = nx.Graph()
        G1.add_weighted_edges_from([
            ('a', 'b', 0.5),
            ('b', 'c', 1.0),
            ('a', 'c', 2.0),
        ])
        Q1 = dnx.traveling_salesperson_qubo(G1, lagrange=10)

        G2 = nx.Graph()
        G2.add_weighted_edges_from([
            ('a', 'b', 0.5),
            ('a', 'c', 2.0),
        ])
        # make sure that missing_edge_weight gets applied correctly
        Q2 = dnx.traveling_salesperson_qubo(G2, lagrange=10, missing_edge_weight=1.0)

        self.assertDictEqual(Q1, Q2)

    def test_digraph_missing_edges(self):
        G1 = nx.DiGraph()
        G1.add_weighted_edges_from([
            ('a', 'b', 0.5),
            ('b', 'a', 0.8),
            ('b', 'c', 1.0),
            ('c', 'b', 0.7),
            ('a', 'c', 2.0),
            ('c', 'a', 2.0),
        ])
        Q1 = dnx.traveling_salesperson_qubo(G1, lagrange=10)

        G2 = nx.DiGraph()
        G2.add_weighted_edges_from([
            ('a', 'b', 0.5),
            ('b', 'a', 0.8),
            ('c', 'b', 0.7),
            ('a', 'c', 2.0),
            ('c', 'a', 2.0),
        ])

        # make sure that missing_edge_weight gets applied correctly
        Q2 = dnx.traveling_salesperson_qubo(G2, lagrange=10, missing_edge_weight=1.0)

        self.assertDictEqual(Q1, Q2)

    def test_k4_equal_weights(self):
        # k5 with all equal weights so all paths are equally good
        G = nx.Graph()
        G.add_weighted_edges_from((u, v, .5)
                                  for u, v in itertools.combinations(range(4), 2))

        Q = dnx.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # all routes are min weight
        min_routes = list(itertools.permutations(G.nodes))

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm.variables}
            for idx, city in enumerate(route):
                sample[(city, idx)] = 1
            self.assertAlmostEqual(bqm.energy(sample), ground_energy)

        # all min-energy solutions are valid routes
        ground_count = 0
        for sample, energy in sampleset.data(['sample', 'energy']):
            if abs(energy - ground_energy) > .001:
                break
            ground_count += 1

        self.assertEqual(ground_count, len(min_routes))

    def test_k4(self):
        # good routes are 0,1,2,3 or 3,2,1,0 (and their rotations)
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1),
                                   (1, 2, 1),
                                   (2, 3, 1),
                                   (3, 0, 1),
                                   (0, 2, 2),
                                   (1, 3, 2)])

        Q = dnx.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # good routes won't have 0<->2 or 1<->3
        min_routes = [(0, 1, 2, 3),
                      (1, 2, 3, 0),
                      (2, 3, 0, 1),
                      (1, 2, 3, 0),
                      (3, 2, 1, 0),
                      (2, 1, 0, 3),
                      (1, 0, 3, 2),
                      (0, 3, 2, 1)]

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm.variables}
            for idx, city in enumerate(route):
                sample[(city, idx)] = 1
            self.assertAlmostEqual(bqm.energy(sample), ground_energy)

        # all min-energy solutions are valid routes
        ground_count = 0
        for sample, energy in sampleset.data(['sample', 'energy']):
            if abs(energy - ground_energy) > .001:
                break
            ground_count += 1

        self.assertEqual(ground_count, len(min_routes))

    def test_weighted_complete_graph(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(0, 1, 1), (0, 2, 100),
                                   (0, 3, 1), (1, 2, 1),
                                   (1, 3, 100), (2, 3, 1)})

        lagrange = 5.0

        Q = dnx.traveling_salesperson_qubo(G, lagrange, 'weight')

        N = G.number_of_nodes()
        correct_sum = G.size('weight')*2*N-2*N*N*lagrange+2*N*N*(N-1)*lagrange

        actual_sum = sum(Q.values())

        self.assertEqual(correct_sum, actual_sum)

    def test_exceptions(self):
        G = nx.Graph([(0, 1)])
        with self.assertRaises(ValueError):
            dnx.traveling_salesperson_qubo(G)

    def test_docstring_size(self):
        # in the docstring we state the size of the resulting BQM, this checks
        # that
        for n in range(3, 20):
            G = nx.Graph()
            G.add_weighted_edges_from((u, v, .5)
                                      for u, v
                                      in itertools.combinations(range(n), 2))
            Q = dnx.traveling_salesperson_qubo(G)
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

            self.assertEqual(len(bqm), n**2)
            self.assertEqual(len(bqm.quadratic), 2*n*n*(n - 1))
