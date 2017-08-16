import unittest

import networkx as nx
import dwave_networkx as dnx


class TestHeuristic:
    def test_basic(self):
        G = nx.Graph()

        tw, order = self.heuristic(G)

        self.assertGreaterEqual(tw, 0)
        self.check_order(G, order)

    def test_complete(self):
        G = nx.complete_graph(10)

        tw, order = self.heuristic(G)
        self.assertGreaterEqual(tw, 9)
        self.check_order(G, order)

    def test_chimera(self):
        G = dnx.chimera_graph(2, 2, 4)

        tw, order = self.heuristic(G)
        self.assertGreaterEqual(tw, 8)
        self.check_order(G, order)

    def test_cycle(self):
        G = nx.cycle_graph(43)

        tw, order = self.heuristic(G)
        self.assertGreaterEqual(tw, 2)
        self.check_order(G, order)

    def test_grid(self):
        G = nx.grid_2d_graph(6, 7)

        tw, order = self.heuristic(G)
        self.assertGreaterEqual(tw, 6)
        self.check_order(G, order)

    def check_order(self, G, order):
        self.assertEqual(set(G), set(order))


class TestMinWidth(unittest.TestCase, TestHeuristic):
    def setUp(self):
        self.heuristic = dnx.min_width_heuristic


class TestMinFill(unittest.TestCase, TestHeuristic):
    def setUp(self):
        self.heuristic = dnx.min_fill_heuristic


class TestMaxCardinality(unittest.TestCase, TestHeuristic):
    def setUp(self):
        self.heuristic = dnx.max_cardinality_heuristic


class TestMinorMinWidth(unittest.TestCase):
    def test_basic(self):
        G = nx.complete_graph(10)
        lb = dnx.minor_min_width(G)
        self.assertLessEqual(lb, len(G) - 1)
        self.assertEqual(len(G), 10)


class TestSimplicialTests(unittest.TestCase):
    def test_typical(self):

        # every node in a complete graph is simplicial
        G = nx.complete_graph(100)
        for v in G:
            self.assertTrue(dnx.is_simplicial(G, v))

        # if we remove one edge then every node except the
        # two should be almost simplicial
        u = 0
        w = 1
        G.remove_edge(u, w)
        for v in G:
            if v not in (u, w):
                self.assertFalse(dnx.is_simplicial(G, v))
                self.assertTrue(dnx.is_almost_simplicial(G, v))
            else:
                self.assertTrue(dnx.is_simplicial(G, v))


class TestBranchAndBound(unittest.TestCase):
    def test_empty(self):
        G = nx.Graph()
        true_tw = 0

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def test_chimera(self):
        G = dnx.chimera_graph(1, 1, 4)
        true_tw = 4

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

        G = dnx.chimera_graph(2, 2, 3)
        true_tw = 6

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

        G = dnx.chimera_graph(1, 2, 4)
        true_tw = 4

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

        G = dnx.chimera_graph(2, 2, 4)
        true_tw = 8

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def test_complete(self):
        G = nx.complete_graph(100)
        true_tw = 99

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def test_complete_minus_1(self):
        G = nx.complete_graph(50)
        G.remove_edge(7, 6)
        true_tw = 48

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def test_cycle(self):
        G = nx.cycle_graph(167)
        true_tw = 2

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def test_grid(self):
        G = nx.grid_2d_graph(4, 6)
        true_tw = 4

        tw, order = dnx.treewidth_branch_and_bound(G)
        self.check_order(G, order)
        self.assertEqual(tw, true_tw)

    def check_order(self, G, order):
        self.assertEqual(set(G), set(order))
