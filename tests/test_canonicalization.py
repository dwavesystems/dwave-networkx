# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import dimod
import dwave.graphs as dnx
import networkx as nx


class TestRootedTile(unittest.TestCase):
    def test_C33_tiles(self):
        # not a public function
        rooted_tile = dnx.algorithms.canonicalization.rooted_tile

        C33 = dnx.chimera_graph(3, 3, 4)

        for root in range(0, len(C33), 8):

            horiz, vert = rooted_tile(C33.adj, root, 4)

            self.assertEqual(horiz, set(range(root, root+4)))
            self.assertEqual(vert, set(range(root+4, root+8)))


class TestCanonicalChimeraLabeling(unittest.TestCase):
    def test_tile_identity(self):
        C1 = dnx.chimera_graph(1)
        coord = dnx.chimera_coordinates(1, 1, 4)

        labels = dnx.canonical_chimera_labeling(C1)
        labels = {v: coord.chimera_to_linear(labels[v]) for v in labels}

        G = nx.relabel_nodes(C1, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C1))
        self.assertEqual(set(G), set(C1))

    def test_bqm_tile_identity(self):
        J = {e: -1 for e in dnx.chimera_graph(1).edges}
        C1bqm = dimod.BinaryQuadraticModel.from_ising({}, J)
        coord = dnx.chimera_coordinates(1, 1, 4)

        labels = dnx.canonical_chimera_labeling(C1bqm)
        labels = {v: coord.chimera_to_linear(labels[v]) for v in labels}

        bqm = C1bqm.relabel_variables(labels, inplace=False)

        self.assertEqual(bqm, C1bqm)

    def test_row_identity(self):
        C41 = dnx.chimera_graph(4, 1)
        coord = dnx.chimera_coordinates(4, 1, 4)

        labels = dnx.canonical_chimera_labeling(C41)
        labels = {v: coord.chimera_to_linear(labels[v]) for v in labels}

        G = nx.relabel_nodes(C41, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C41))

    def test_3x3_identity(self):
        C33 = dnx.chimera_graph(3, 3)
        coord = dnx.chimera_coordinates(3, 3, 4)

        labels = dnx.canonical_chimera_labeling(C33)
        labels = {v: coord.chimera_to_linear(labels[v]) for v in labels}

        G = nx.relabel_nodes(C33, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C33))

    def test_construction_string_labels(self):
        C22 = dnx.chimera_graph(2, 2, 3)
        coord = dnx.chimera_coordinates(2, 2, 3)

        alpha = 'abcdefghijklmnopqrstuvwxyz'

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        for u, v in reversed(list(C22.edges)):
            bqm.add_interaction(alpha[u], alpha[v], 1)

        assert len(bqm.quadratic) == len(C22.edges)
        assert len(bqm) == len(C22)

        labels = dnx.canonical_chimera_labeling(bqm)
        labels = {v: alpha[coord.chimera_to_linear(labels[v])] for v in labels}

        bqm2 = bqm.relabel_variables(labels, inplace=False)

        self.assertEqual(bqm, bqm2)

    def test_reversed(self):
        C33 = nx.Graph() #Ordering is guaranteed Python>=3.7, OrderedGraph is deprecated
        C33.add_nodes_from(reversed(range(3*3*4)))
        C33.add_edges_from(dnx.chimera_graph(3, 3, 4).edges)
        coord = dnx.chimera_coordinates(3, 3, 4)

        labels = dnx.canonical_chimera_labeling(C33)
        labels = {v: coord.chimera_to_linear(labels[v]) for v in labels}

        G = nx.relabel_nodes(C33, labels, copy=True)

        self.assertTrue(nx.is_isomorphic(G, C33))

    def test__shore_size_tiles(self):
        shore_size = dnx.algorithms.canonicalization._chimera_shore_size

        for t in range(1, 8):
            G = dnx.chimera_graph(1, 1, t)
            self.assertEqual(shore_size(G.adj, len(G.edges)), t)

    def test__shore_size_columns(self):
        shore_size = dnx.algorithms.canonicalization._chimera_shore_size

        # 2, 1, 1 is the same as 1, 1, 2
        for m in range(2, 11):
            for t in range(9, 1, -1):
                G = dnx.chimera_graph(m, 1, t)
                self.assertEqual(shore_size(G.adj, len(G.edges)), t)

    def test__shore_size_rectangles(self):
        shore_size = dnx.algorithms.canonicalization._chimera_shore_size

        # 2, 1, 1 is the same as 1, 1, 2
        for m in range(2, 7):
            for n in range(2, 7):
                for t in range(1, 6):
                    G = dnx.chimera_graph(m, n, t)
                    self.assertEqual(shore_size(G.adj, len(G.edges)), t)
