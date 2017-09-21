import unittest
import itertools

import networkx as nx
import dwave_networkx as dnx
from dwave_networkx.utils.test_samplers import ExactSolver, FastSampler, qubo_energy

from dwave_networkx.algorithms.coloring import _vertex_different_colors_qubo
from dwave_networkx.algorithms.coloring import _vertex_one_color_qubo
from dwave_networkx.algorithms.coloring import _minimum_coloring_qubo

try:
    import numpy
    _numpy = True
except ImportError:
    _numpy = False


class TestColor(unittest.TestCase):

    def test__vertex_different_colors_qubo(self):
        # Chimera tile (can be 2-colored)
        G = dnx.chimera_graph(1, 1, 4)
        counter = itertools.count()
        x_vars = {v: {0: next(counter), 1: next(counter)} for v in G}

        # get the QUBO
        Q = _vertex_different_colors_qubo(G, x_vars)

        # this thing should have energy 0 when each node is a different color
        bicolor = {v: 0 for v in range(4)}
        bicolor.update({v: 1 for v in range(4, 8)})

        # make the sample from the bicolor
        sample = {}
        for v in G:
            if bicolor[v] == 0:
                sample[x_vars[v][0]] = 1
                sample[x_vars[v][1]] = 0
            else:
                sample[x_vars[v][0]] = 0
                sample[x_vars[v][1]] = 1

        self.assertEqual(qubo_energy(Q, sample), 0)

    def test__vertex_one_color_qubo(self):
        G = dnx.chimera_graph(2, 2, 4)
        counter = itertools.count()
        x_vars = {v: {0: next(counter), 1: next(counter)} for v in G}

        # get the QUBO
        Q = _vertex_one_color_qubo(x_vars)

        # assign each variable a single color
        sample = {}
        for v in G:
            sample[x_vars[v][0]] = 1
            sample[x_vars[v][1]] = 0

        self.assertEqual(qubo_energy(Q, sample), -1 * len(G))

    def test__minimum_coloring_qubo(self):
        # Chimera tile (can be 2-colored)
        G = dnx.chimera_graph(1, 1, 4)
        chi_ub = 5
        chi_lb = 2
        possible_colors = {v: set(range(chi_ub)) for v in G}

        counter = itertools.count()
        x_vars = {v: {c: next(counter) for c in possible_colors[v]} for v in G}

        # get the QUBO
        Q = _minimum_coloring_qubo(x_vars, chi_lb, chi_ub)

        # TODO: actually test something other than it running

    def test_vertex_color_basic(self):
        # all small enough for an exact solver to handle them reasonably
        G = dnx.chimera_graph(1, 2, 2)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

        G = nx.path_graph(5)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

        for __ in range(10):
            G = nx.gnp_random_graph(5, .5)
            coloring = dnx.min_vertex_coloring(G, ExactSolver())
            self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_complete_graph(self):
        # this should get eliminated in software so be fast to run
        G = nx.complete_graph(101)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_odd_cycle_graph(self):
        """Graph that is an odd circle"""
        G = nx.cycle_graph(5)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_no_edge_graph(self):
        """Graph with many nodes but no edges, should be caught before QUBO"""
        # this should get eliminated in software so be fast to run
        G = nx.Graph()
        G.add_nodes_from(range(100))
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_disconnected_graph(self):
        """One edge and one disconnected node"""
        G = nx.path_graph(2)
        G.add_node(3)

        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_disconnected_cycle_graph(self):
        G = nx.complete_graph(3)  # odd 3-cycle
        G.add_node(4)  # floating node
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_vertex_color_almost_complete(self):
        # this should get eliminated in software so be fast to run
        G = nx.complete_graph(10)
        mapping = dict(zip(G.nodes(), "abcdefghijklmnopqrstuvwxyz"))
        G = nx.relabel_nodes(G, mapping)
        n0, n1 = next(iter(G.edges()))
        G.remove_edge(n0, n1)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    @unittest.skipIf(FastSampler is None, "no FastSampler to test with")
    def test_dimod_response_vs_list(self):
        # should be able to handle either a dimod response or a list of dicts
        G = dnx.chimera_graph(1, 1, 3)
        coloring = dnx.min_vertex_coloring(G, ExactSolver())
        coloring = dnx.min_vertex_coloring(G, FastSampler())
