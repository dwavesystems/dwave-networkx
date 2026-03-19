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

import dimod

import dwave.graphs as dnx


class TestMinVertexColor(unittest.TestCase):

    def test_5path(self):
        G = nx.path_graph(5)
        coloring = dnx.min_vertex_coloring(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))
        self.assertEqual(len(set(coloring.values())), 2)  # bipartite

    def test_odd_cycle_graph(self):
        """Graph that is an odd circle"""
        G = nx.cycle_graph(5)
        coloring = dnx.min_vertex_coloring(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_disconnected_graph(self):
        """One edge and one disconnected node"""
        G = nx.path_graph(2)
        G.add_node(3)

        coloring = dnx.min_vertex_coloring(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_disconnected_cycle_graph(self):
        G = nx.complete_graph(3)  # odd 3-cycle
        G.add_node(4)  # floating node
        coloring = dnx.min_vertex_coloring(G, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))


class TestMinVertexColorQubo(unittest.TestCase):
    def test_chromatic_number(self):
        G = nx.cycle_graph('abcd')

        # when the chromatic number is fixed this is exactly vertex_color
        self.assertEqual(dnx.min_vertex_color_qubo(G, chromatic_lb=2,
                                                   chromatic_ub=2),
                         dnx.vertex_color_qubo(G, 2))


class TestVertexColor(unittest.TestCase):
    def test_4cycle(self):
        G = nx.cycle_graph('abcd')

        coloring = dnx.vertex_color(G, 2, dimod.ExactSolver())

        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_4cycle_with_chord(self):
        G = nx.cycle_graph(4)
        G.add_edge(0, 2)

        # need 3 colors in this case
        coloring = dnx.vertex_color(G, 3, dimod.ExactSolver())

        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_5cycle(self):
        G = nx.cycle_graph(5)
        coloring = dnx.vertex_color(G, 3, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))

    def test_disconnected_cycle_graph(self):
        G = nx.complete_graph(3)  # odd 3-cycle
        G.add_node(4)  # floating node
        coloring = dnx.vertex_color(G, 3, dimod.ExactSolver())
        self.assertTrue(dnx.is_vertex_coloring(G, coloring))


class TestVertexColorQUBO(unittest.TestCase):
    def test_single_node(self):
        G = nx.Graph()
        G.add_node('a')

        # a single color
        Q = dnx.vertex_color_qubo(G, ['red'])

        self.assertEqual(Q, {(('a', 'red'), ('a', 'red')): -1})

    def test_4cycle(self):
        G = nx.cycle_graph('abcd')

        Q = dnx.vertex_color_qubo(G, 2)

        sampleset = dimod.ExactSolver().sample_qubo(Q)

        # check that the ground state is a valid coloring
        ground_energy = sampleset.first.energy

        colorings = []
        for sample, en in sampleset.data(['sample', 'energy']):
            if en > ground_energy:
                break

            coloring = {}
            for (v, c), val in sample.items():
                if val:
                    coloring[v] = c

            self.assertTrue(dnx.is_vertex_coloring(G, coloring))

            colorings.append(coloring)

        # there are two valid colorings
        self.assertEqual(len(colorings), 2)

        self.assertEqual(ground_energy, -len(G))

    def test_num_variables(self):
        G = nx.Graph()
        G.add_nodes_from(range(15))

        Q = dnx.vertex_color_qubo(G, 7)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        self.assertEqual(len(bqm.quadratic), len(G)*7*(7-1)/2)

        # add one edge
        G.add_edge(0, 1)
        Q = dnx.vertex_color_qubo(G, 7)
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        self.assertEqual(len(bqm.quadratic), len(G)*7*(7-1)/2 + 7)

    def test_docstring_stats(self):
        # get a complex-ish graph
        G = nx.karate_club_graph()

        colors = range(10)

        Q = dnx.vertex_color_qubo(G, colors)

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        self.assertEqual(len(bqm), len(G)*len(colors))
        self.assertEqual(len(bqm.quadratic), len(G)*len(colors)*(len(colors)-1)/2
                         + len(G.edges)*len(colors))
