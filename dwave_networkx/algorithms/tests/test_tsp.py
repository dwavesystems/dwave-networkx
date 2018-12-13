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
        for u,v in G.edges(): G[u][v]['weight']=1
        route = tsp.traveling_salesman(G, dimod.ExactSolver())
        self.assertTrue(tsp.is_hamiltonian_path(G, route))

        G = nx.complete_graph(4)
        for u,v in G.edges(): G[u][v]['weight']=u+v
        route = tsp.traveling_salesman(G, dimod.ExactSolver(), lagrange=10.0)
        self.assertTrue(tsp.is_hamiltonian_path(G, route))

    def test_dimod_vs_list(self):
        G = nx.complete_graph(4)
        for u,v in G.edges(): G[u][v]['weight']=1

        route = tsp.traveling_salesman(G, dimod.ExactSolver())
        route = tsp.traveling_salesman(G, dimod.SimulatedAnnealingSampler())
