import unittest

import networkx as nx
import dwave_networkx as dnx


class TestMinWidth(unittest.TestCase):
    def test_empty(self):
        G = nx.empty_graph()

        tw, order = dnx.min_width_heuristic(G)
        self.assertEqual(tw, 0)
        self.assertEqual(order, [])
        self.assertEqual(len(order), len(set(order)))

    def test_complete(self):
        G = nx.complete_graph(10)

        tw, order = dnx.min_width_heuristic(G)
        self.assertEqual(tw, 9)
        self.assertEqual(len(order), len(G))
        for v in order:
            self.assertIn(v, G)
        self.assertEqual(len(order), len(set(order)))

    def test_inplace(self):
        G = nx.complete_graph(10)
        tw, order = dnx.min_width_heuristic(G, inplace=True)
        self.assertEqual(len(G), 0)
        self.assertEqual(len(order), len(set(order)))


class TestMinFill(unittest.TestCase):
    def test_empty(self):
        G = nx.empty_graph()

        tw, order = dnx.min_fill_heuristic(G)
        self.assertEqual(tw, 0)
        self.assertEqual(order, [])
        self.assertEqual(len(order), len(set(order)))

    def test_complete(self):
        G = nx.complete_graph(10)

        tw, order = dnx.min_fill_heuristic(G)
        self.assertEqual(tw, 9)
        self.assertEqual(len(order), len(G))
        for v in order:
            self.assertIn(v, G)
        self.assertEqual(len(order), len(set(order)))

    def test_inplace(self):
        G = nx.complete_graph(10)
        tw, order = dnx.min_fill_heuristic(G, inplace=True)
        self.assertEqual(len(G), 0)
        self.assertEqual(len(order), len(set(order)))


class TestMaxCardinality(unittest.TestCase):
    def test_empty(self):
        G = nx.empty_graph()

        tw, order = dnx.max_cardinality_heuristic(G)
        self.assertEqual(tw, 0)
        self.assertEqual(order, [])
        self.assertEqual(len(order), len(set(order)))

    def test_complete(self):
        G = nx.complete_graph(10)

        tw, order = dnx.max_cardinality_heuristic(G)
        self.assertEqual(tw, 9)
        self.assertEqual(len(order), len(G))
        for v in order:
            self.assertIn(v, G)
        self.assertEqual(len(order), len(set(order)))

    def test_inplace(self):
        G = nx.complete_graph(10)
        tw, order = dnx.max_cardinality_heuristic(G, inplace=True)
        self.assertEqual(len(G), 0)
        self.assertEqual(len(order), len(set(order)))


class TestMinorMinWidth(unittest.TestCase):
    def test_basic(self):
        G = nx.complete_graph(10)
        lb = dnx.minor_min_width(G)
        self.assertLessEqual(lb, len(G) - 1)
        self.assertEqual(len(G), 10)


class TestSimplicialTests(unittest.TestCase):
    def test_is_simplicial(self):

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
