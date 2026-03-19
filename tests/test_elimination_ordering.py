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

import unittest

import networkx as nx
import dwave.graphs as dnx


class HeuristicCases:
    """Change the name for compatibility with nose."""

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

    def test_disjoint(self):
        graph = nx.complete_graph(1)
        graph.add_node(1)

        tw, order = self.heuristic(graph)
        self.assertGreaterEqual(tw, 0)
        self.check_order(graph, order)

        graph = nx.complete_graph(4)
        graph.add_node(4)

        tw, order = self.heuristic(graph)
        self.assertGreaterEqual(tw, 3)
        self.check_order(graph, order)

        graph = nx.complete_graph(4)
        graph.add_edge(4, 5)

        tw, order = self.heuristic(graph)
        self.assertGreaterEqual(tw, 3)
        self.check_order(graph, order)

    def test_self_loop(self):
        graph = nx.complete_graph(3)
        graph.add_edge(0, 0)
        graph.add_edge(2, 2)

        tw, order = self.heuristic(graph)
        self.assertGreaterEqual(tw, 2)
        self.check_order(graph, order)


class TestMinWidth(unittest.TestCase, HeuristicCases):
    def setUp(self):
        self.heuristic = dnx.min_width_heuristic


class TestMinFill(unittest.TestCase, HeuristicCases):
    def setUp(self):
        self.heuristic = dnx.min_fill_heuristic


class TestMaxCardinality(unittest.TestCase, HeuristicCases):
    def setUp(self):
        self.heuristic = dnx.max_cardinality_heuristic


class TestMinorMinWidth(unittest.TestCase):
    def test_basic(self):
        G = nx.complete_graph(10)
        lb = dnx.minor_min_width(G)
        self.assertLessEqual(lb, len(G) - 1)
        self.assertEqual(len(G), 10)

    def test_disjoint(self):
        graph = nx.complete_graph(1)
        graph.add_node(1)

        lb = dnx.minor_min_width(graph)
        self.assertEqual(lb, 0)

        graph = nx.complete_graph(4)
        graph.add_node(4)

        lb = dnx.minor_min_width(graph)
        self.assertEqual(lb, 3)

        graph = nx.complete_graph(4)
        graph.add_edge(4, 5)

        lb = dnx.minor_min_width(graph)
        self.assertEqual(lb, 3)

    def test_self_loop(self):
        graph = nx.complete_graph(3)
        graph.add_edge(0, 0)
        graph.add_edge(2, 2)

        lb = dnx.minor_min_width(graph)


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

    def test_disjoint(self):
        graph = nx.complete_graph(1)
        graph.add_node(1)

        tw, order = dnx.treewidth_branch_and_bound(graph)
        self.check_order(graph, order)
        self.assertEqual(tw, 0)

        graph = nx.complete_graph(4)
        graph.add_node(4)

        tw, order = dnx.treewidth_branch_and_bound(graph)
        self.check_order(graph, order)
        self.assertEqual(tw, 3)

        graph = nx.complete_graph(4)
        graph.add_edge(4, 5)

        tw, order = dnx.treewidth_branch_and_bound(graph)
        self.check_order(graph, order)
        self.assertEqual(tw, 3)

    def test_upperbound_parameter(self):
        """We want to be able to give an upper bound to make execution faster."""
        graph = nx.complete_graph(3)

        # providing a treewidth without an order should still work, although
        # the order might not be found
        tw, order = dnx.treewidth_branch_and_bound(graph, treewidth_upperbound=2)
        self.assertEqual(len(order), 3)  # all nodes should be in the order

        tw, order = dnx.treewidth_branch_and_bound(graph, [0, 1, 2])
        self.assertEqual(len(order), 3)  # all nodes should be in the order

        # try with both
        tw, order = dnx.treewidth_branch_and_bound(graph, [0, 1, 2], 2)
        self.assertEqual(len(order), 3)  # all nodes should be in the order

    def test_incorrect_lowerbound(self):
        graph = nx.complete_graph(3)

        tw, order = dnx.treewidth_branch_and_bound(graph, treewidth_upperbound=1)
        self.assertEqual(order, [])  # no order produced

    def test_singleton(self):
        G = nx.Graph()
        G.add_node('a')

        tw, order = dnx.treewidth_branch_and_bound(G)

    def test_self_loop(self):
        graph = nx.complete_graph(3)
        graph.add_edge(0, 0)
        graph.add_edge(2, 2)

        tw, order = dnx.treewidth_branch_and_bound(graph)

        self.assertEqual(tw, 2)


class TestEliminationOrderWidth(unittest.TestCase):
    def test_trivial(self):
        G = nx.Graph()
        tw = dnx.elimination_order_width(G, [])
        self.assertEqual(tw, 0)

    def test_graphs(self):

        H = nx.complete_graph(2)
        H.add_edge(2, 3)

        graphs = [nx.complete_graph(7),
                  dnx.chimera_graph(2, 1, 3),
                  nx.balanced_tree(5, 3),
                  nx.barbell_graph(8, 11),
                  nx.cycle_graph(5),
                  H]

        for G in graphs:
            tw, order = dnx.treewidth_branch_and_bound(G)
            self.assertEqual(dnx.elimination_order_width(G, order), tw)

            tw, order = dnx.min_width_heuristic(G)
            self.assertEqual(dnx.elimination_order_width(G, order), tw)

            tw, order = dnx.min_fill_heuristic(G)
            self.assertEqual(dnx.elimination_order_width(G, order), tw)

            tw, order = dnx.max_cardinality_heuristic(G)
            self.assertEqual(dnx.elimination_order_width(G, order), tw)

    def test_exceptions(self):

        G = nx.complete_graph(6)
        order = range(4)

        with self.assertRaises(ValueError):
            dnx.elimination_order_width(G, order)

        order = range(7)
        with self.assertRaises(ValueError):
            dnx.elimination_order_width(G, order)


class TestChimeraEliminationOrder(unittest.TestCase):
    def test_variable_order(self):
        n = 8
        m = 10
        p = dnx.chimera_graph(n, m)
        o = dnx.chimera_elimination_order(n, m)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 4*n)

        p = dnx.chimera_graph(m, n)
        o = dnx.chimera_elimination_order(m, n)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 4*n)


class TestPegasusEliminationOrder(unittest.TestCase):
    def test_variable_order(self):
        n = 4
        p = dnx.pegasus_graph(n, fabric_only=False)
        o = dnx.pegasus_elimination_order(n)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 12*n-4)

        p = dnx.pegasus_graph(n, fabric_only=False, coordinates=True)
        o = dnx.pegasus_elimination_order(n, coordinates=True)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 12*n-4)
