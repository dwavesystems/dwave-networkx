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
#
# =============================================================================
import itertools
import unittest

import networkx as nx

import dimod

import dwave_networkx as dnx
import dwave_networkx.algorithms.tsp as tsp


class TestIsHamiltonCycle(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()

        self.assertTrue(tsp.is_hamiltonian_path(G, []))

    def test_K1(self):
        G = nx.complete_graph(1)

        self.assertTrue(tsp.is_hamiltonian_path(G, [0]))
        self.assertFalse(tsp.is_hamiltonian_path(G, []))

    def test_K2(self):
        G = nx.complete_graph(2)

        self.assertTrue(tsp.is_hamiltonian_path(G, [0, 1]))
        self.assertTrue(tsp.is_hamiltonian_path(G, [1, 0]))
        self.assertFalse(tsp.is_hamiltonian_path(G, [0]))
        self.assertFalse(tsp.is_hamiltonian_path(G, [1]))
        self.assertFalse(tsp.is_hamiltonian_path(G, []))

    def test_K3(self):
        G = nx.complete_graph(3)

        self.assertTrue(tsp.is_hamiltonian_path(G, [0, 1, 2]))
        self.assertTrue(tsp.is_hamiltonian_path(G, [1, 0, 2]))
        self.assertFalse(tsp.is_hamiltonian_path(G, [0, 1]))
        self.assertFalse(tsp.is_hamiltonian_path(G, [0]))
        self.assertFalse(tsp.is_hamiltonian_path(G, [1]))
        self.assertFalse(tsp.is_hamiltonian_path(G, []))


class TestTSP(unittest.TestCase):

    def test_TSP_basic(self):
        """Runs the function on some small and simple graphs, just to make
        sure it works in basic functionality.
        """
        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        route = tsp.traveling_salesperson(G, dimod.ExactSolver())
        self.assertTrue(tsp.is_hamiltonian_path(G, route))

        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = u+v
        route = tsp.traveling_salesperson(G, dimod.ExactSolver(), lagrange=10.0)
        self.assertTrue(tsp.is_hamiltonian_path(G, route))

    def test_dimod_vs_list(self):
        G = nx.complete_graph(4)
        for u, v in G.edges():
            G[u][v]['weight'] = 1

        route = tsp.traveling_salesperson(G, dimod.ExactSolver())
        route = tsp.traveling_salesperson(G, dimod.SimulatedAnnealingSampler())

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

        route = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=2)

        self.assertEqual(route[0], 2)


class TestTSPQUBO(unittest.TestCase):
    def test_empty(self):
        Q = tsp.traveling_salesperson_qubo(nx.Graph())
        self.assertEqual(Q, {})

    def test_k3(self):
        # 3cycle so all paths are equally good
        G = nx.Graph()
        G.add_weighted_edges_from([('a', 'b', 0.5),
                                   ('b', 'c', 1.0),
                                   ('a', 'c', 2.0)])

        Q = tsp.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # all routes are min weight
        min_routes = list(itertools.permutations(G.nodes))

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm}
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

    def test_k4_equal_weights(self):
        # k5 with all equal weights so all paths are equally good
        G = nx.Graph()
        G.add_weighted_edges_from((u, v, .5)
                                  for u, v in itertools.combinations(range(4), 2))

        Q = tsp.traveling_salesperson_qubo(G, lagrange=10)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        # all routes are min weight
        min_routes = list(itertools.permutations(G.nodes))

        # get the min energy of the qubo
        sampleset = dimod.ExactSolver().sample(bqm)
        ground_energy = sampleset.first.energy

        # all possible routes are equally good
        for route in min_routes:
            sample = {v: 0 for v in bqm}
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

        Q = tsp.traveling_salesperson_qubo(G, lagrange=10)
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
            sample = {v: 0 for v in bqm}
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

    def test_exceptions(self):
        G = nx.Graph([(0, 1)])
        with self.assertRaises(ValueError):
            tsp.traveling_salesperson_qubo(G)

    def test_docstring_size(self):
        # in the docstring we state the size of the resulting BQM, this checks
        # that
        for n in range(3, 20):
            G = nx.Graph()
            G.add_weighted_edges_from((u, v, .5)
                                      for u, v
                                      in itertools.combinations(range(n), 2))
            Q = tsp.traveling_salesperson_qubo(G)
            bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

            self.assertEqual(len(bqm), n**2)
            self.assertEqual(len(bqm.quadratic), 2*n*n*(n - 1))
