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
# ================================================================================================
from __future__ import division

import unittest
import warnings

import networkx as nx
import dwave_networkx as dnx
from dwave_networkx.generators.pegasus import (pegasus_graph, get_tuple_fragmentation_fn,
                                               get_tuple_defragmentation_fn)

alpha_map = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))


class TestPegasusGraph(unittest.TestCase):
    def test_p2(self):
        G = dnx.pegasus_graph(2, fabric_only=False)

        # should have 48 nodes
        self.assertEqual(len(G), 48)

        # nodes 0,...,47 should be in the graph
        for n in range(48):
            self.assertIn(n, G)

    def test_bad_args(self):
        with self.assertRaises(dnx.DWaveNetworkXException):
            G = dnx.pegasus_graph(2, offset_lists=[], offsets_index=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            G = dnx.pegasus_graph(2, offset_lists=[[0, 1]*6, [0, 1]*6])
            self.assertLessEqual(len(w), 13)
            self.assertGreaterEqual(len(w), 12)

    def test_connected_component(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        test_offsets = [[0] * 12] * 2, [[2] * 12, [6] * 12], [[6] * 12, [2, 2, 6, 6, 10, 10] * 2], [[2, 2, 6, 6, 10, 10] * 2] * 2
        for offsets in test_offsets:
            G = dnx.pegasus_graph(4, fabric_only=True, offset_lists=offsets)
            H = dnx.pegasus_graph(4, fabric_only=False, offset_lists=offsets)
            nodes = sorted(G)
            comp = sorted(max((G.subgraph(c).copy() for c in nx.connected_components(G)), key=len))
            self.assertEqual(comp, nodes)

    def test_coordinate_basics(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        G = dnx.pegasus_graph(4, fabric_only=False)
        H = dnx.pegasus_graph(4, coordinates=True, fabric_only=False)
        coords = pegasus_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['pegasus_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

    def test_coordinate_subgraphs(self):
        from dwave_networkx.generators.pegasus import pegasus_coordinates
        from random import sample
        G = dnx.pegasus_graph(4)
        H = dnx.pegasus_graph(4, coordinates=True)
        coords = pegasus_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.tuples(lmask))

        self.assertEqual(lmask, list(coords.ints(cmask)))

        Gm = dnx.pegasus_graph(4, node_list=lmask)
        Hm = dnx.pegasus_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.pegasus_graph(4, edge_list=EG)
        Hn = dnx.pegasus_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['pegasus_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.int(q))
            self.assertEqual(q, coords.tuple(v))

        self.assertEqual(EG, sorted(map(sorted, coords.int_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.tuple_pairs(Gn.edges()))))

    def test_variable_order(self):
        n = 4
        p = dnx.pegasus_graph(n, fabric_only=False)
        o = dnx.generators.pegasus.pegasus_elimination_order(n)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 12*n-4)

        p = dnx.pegasus_graph(n, fabric_only=False, coordinates=True)
        o = dnx.generators.pegasus.pegasus_elimination_order(n, coordinates=True)
        tw = dnx.elimination_order_width(p, o)
        self.assertEqual(tw, 12*n-4)


class TestTupleFragmentation(unittest.TestCase):
    def test_empty_list(self):
        # Set up fragmentation function
        pg = pegasus_graph(3)
        fragment_tuple = get_tuple_fragmentation_fn(pg)

        # Fragment pegasus coordinates
        fragments = fragment_tuple([])
        self.assertEqual([], fragments)

    def test_single_horizontal_coordinate(self):
        # Set up fragmentation function
        pg = pegasus_graph(2)
        fragment_tuple = get_tuple_fragmentation_fn(pg)

        # Fragment pegasus coordinates
        pegasus_coord = (1, 0, 0, 0)
        fragments = fragment_tuple([pegasus_coord])

        expected_fragments = {(0, 3, 1, 0),
                              (0, 4, 1, 0),
                              (0, 5, 1, 0),
                              (0, 6, 1, 0),
                              (0, 7, 1, 0),
                              (0, 8, 1, 0)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_single_vertical_coordinate(self):
        # Set up fragmentation function
        pg = pegasus_graph(6)
        fragment_tuple = get_tuple_fragmentation_fn(pg)

        pegasus_coord = (0, 1, 3, 1)
        fragments = fragment_tuple([pegasus_coord])

        expected_fragments = {(7, 7, 0, 1),
                              (8, 7, 0, 1),
                              (9, 7, 0, 1),
                              (10, 7, 0, 1),
                              (11, 7, 0, 1),
                              (12, 7, 0, 1)}

        self.assertEqual(expected_fragments, set(fragments))

    def test_list_of_coordinates(self):
        # Set up fragmentation function
        pg = pegasus_graph(6)
        fragment_tuple = get_tuple_fragmentation_fn(pg)

        # Fragment pegasus coordinates
        pegasus_coords = [(1, 5, 11, 4), (0, 2, 2, 3)]
        fragments = fragment_tuple(pegasus_coords)

        expected_fragments = {(35, 29, 1, 1),
                              (35, 30, 1, 1),
                              (35, 31, 1, 1),
                              (35, 32, 1, 1),
                              (35, 33, 1, 1),
                              (35, 34, 1, 1),
                              (19, 13, 0, 0),
                              (20, 13, 0, 0),
                              (21, 13, 0, 0),
                              (22, 13, 0, 0),
                              (23, 13, 0, 0),
                              (24, 13, 0, 0)}

        self.assertEqual(expected_fragments, set(fragments))


class TestTupleDefragmentation(unittest.TestCase):
    def test_empty_list(self):
        # Set up defragmentation function
        pg = pegasus_graph(2)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = []
        pegasus_coords = defragment_tuple(chimera_coords)

        self.assertEqual([], pegasus_coords)

    def test_single_fragment(self):
        # Set up defragmentation function
        pg = pegasus_graph(4)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(3, 7, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(0, 1, 2, 0)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_multiple_fragments_from_same_qubit(self):
        # Set up defragmentation function
        pg = pegasus_graph(3)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(9, 8, 1, 1), (9, 11, 1, 1)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(1, 1, 7, 1)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_mixed_fragments(self):
        # Set up defragmenation function
        pg = pegasus_graph(8)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(17, 14, 0, 0), (22, 14, 0, 0), (24, 32, 1, 0), (1, 31, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = {(0, 2, 4, 2), (1, 4, 0, 4), (0, 5, 2, 0)}
        self.assertEqual(expected_pegasus_coords, set(pegasus_coords))
