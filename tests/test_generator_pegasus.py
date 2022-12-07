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
import warnings

from random import sample

import networkx as nx
import dwave_networkx as dnx
import numpy as np

from dwave_networkx.generators.pegasus import (
    fragmented_edges,
    get_tuple_defragmentation_fn,
    get_tuple_fragmentation_fn,
    )

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

class TestPegasusTorus(unittest.TestCase):
    def test(self):
        for m in [4]:
            g = dnx.pegasus_torus(m)

            num_var = 24*(m-1)*(m-1)
            self.assertEqual(g.number_of_nodes(),num_var)
            num_edges = (num_var*(13 + (m>2) + (m>3)))//2
            self.assertEqual(g.number_of_edges(),num_edges)
            #(u,w,k,z) -> (u, [w + u dx + (1-u)dy]%(L-1)), k, [z + (1-u) dx + u dy]%(L-1))
            L=m-1 #Cell dimension
            if L>1:
                dx = 1 + np.random.randint(m-1)
                dy = 1 + np.random.randint(m-1)
                relabel = lambda tup: (tup[0],(tup[1] + tup[0]*dx + (1-tup[0])*dy)%L,tup[2],(tup[3] + tup[0]*dy + (1-tup[0])*dx)%L)
                g_translated = nx.relabel_nodes(g,relabel,copy=True)
                g.remove_edges_from(g_translated.edges())
                self.assertEqual(g.number_of_edges(),0)
                g.remove_nodes_from(g_translated.nodes())
                self.assertEqual(g.number_of_nodes(),0)

class TestPegasusCoordinates(unittest.TestCase):

    def test_connected_component(self):
        test_offsets = [[0] * 12] * 2, [[2] * 12, [6] * 12], [[6] * 12, [2, 2, 6, 6, 10, 10] * 2], [[2, 2, 6, 6, 10, 10] * 2] * 2
        for offsets in test_offsets:
            G = dnx.pegasus_graph(4, fabric_only=True, offset_lists=offsets)
            H = dnx.pegasus_graph(4, fabric_only=False, offset_lists=offsets)
            nodes = sorted(G)
            comp = sorted(max((G.subgraph(c).copy() for c in nx.connected_components(G)), key=len))
            self.assertEqual(comp, nodes)

    def test_coordinate_basics(self):
        G = dnx.pegasus_graph(4, fabric_only=False)
        H = dnx.pegasus_graph(4, coordinates=True, fabric_only=False)
        coords = dnx.pegasus_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['pegasus_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.pegasus_to_linear(q))
            self.assertEqual(q, coords.linear_to_pegasus(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.pegasus_to_linear(q))
            self.assertEqual(q, coords.linear_to_pegasus(v))

    def test_nice_coordinates(self):
        G = dnx.pegasus_graph(4, nice_coordinates=True)
        H = dnx.chimera_graph(3, coordinates=True)
        for p, q in H.edges():
            for t in range(3):
                pg = (t,) + p
                qg = (t,) + q
                self.assertTrue(G.has_edge(pg, qg))
        coords = dnx.pegasus_coordinates(4)
        n2p = coords.nice_to_pegasus
        p2n = coords.pegasus_to_nice
        n2l = coords.nice_to_linear
        l2n = coords.linear_to_nice
        for p in G.nodes():
            self.assertEqual(p2n(n2p(p)), p)
            self.assertEqual(l2n(n2l(p)), p)
            self.assertTrue(H.has_node(p[1:]))

        G = dnx.pegasus_graph(4)
        for p in G.nodes():
            self.assertEqual(n2l(l2n(p)), p)

        G = dnx.pegasus_graph(4, coordinates=True)
        for p in G.nodes():
            self.assertEqual(n2p(p2n(p)), p)


    def test_consistent_linear_nice_pegasus(self):
        P4 = dnx.pegasus_graph(4, nice_coordinates=True)

        coords = dnx.pegasus_coordinates(4)

        # `.*_to_*` methods

        for n, data in P4.nodes(data=True):
            p = data['pegasus_index']
            i = data['linear_index']
            self.assertEqual(coords.nice_to_pegasus(n), p)
            self.assertEqual(coords.pegasus_to_nice(p), n)
            self.assertEqual(coords.nice_to_linear(n), i)
            self.assertEqual(coords.linear_to_nice(i), n)
            self.assertEqual(coords.linear_to_pegasus(i), p)
            self.assertEqual(coords.pegasus_to_linear(p), i)

        # `.iter_*_to_*` methods

        nlist, plist, ilist = zip(*((n, d['pegasus_index'], d['linear_index'])
                                    for n, d in P4.nodes(data=True)))
        self.assertEqual(tuple(coords.iter_nice_to_pegasus(nlist)), plist)
        self.assertEqual(tuple(coords.iter_pegasus_to_nice(plist)), nlist)
        self.assertEqual(tuple(coords.iter_nice_to_linear(nlist)), ilist)
        self.assertEqual(tuple(coords.iter_linear_to_nice(ilist)), nlist)
        self.assertEqual(tuple(coords.iter_pegasus_to_linear(plist)), ilist)
        self.assertEqual(tuple(coords.iter_linear_to_pegasus(ilist)), plist)

        # `.iter_*_to_*_pairs` methods

        nedgelist = []
        pedgelist = []
        iedgelist = []
        for u, v in P4.edges:
            nedgelist.append((u, v))
            pedgelist.append((P4.nodes[u]['pegasus_index'],
                              P4.nodes[v]['pegasus_index']))
            iedgelist.append((P4.nodes[u]['linear_index'],
                              P4.nodes[v]['linear_index']))

        self.assertEqual(list(coords.iter_nice_to_pegasus_pairs(nedgelist)),
                         pedgelist)
        self.assertEqual(list(coords.iter_pegasus_to_nice_pairs(pedgelist)),
                         nedgelist)
        self.assertEqual(list(coords.iter_nice_to_linear_pairs(nedgelist)),
                         iedgelist)
        self.assertEqual(list(coords.iter_linear_to_nice_pairs(iedgelist)),
                         nedgelist)
        self.assertEqual(list(coords.iter_pegasus_to_linear_pairs(pedgelist)),
                         iedgelist)
        self.assertEqual(list(coords.iter_linear_to_pegasus_pairs(iedgelist)),
                         pedgelist)

    def test_coordinate_subgraphs(self):
        G = dnx.pegasus_graph(4)
        H = dnx.pegasus_graph(4, coordinates=True)
        coords = dnx.pegasus_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.iter_linear_to_pegasus(lmask))

        self.assertEqual(lmask, list(coords.iter_pegasus_to_linear(cmask)))

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
            self.assertEqual(v, coords.pegasus_to_linear(q))
            self.assertEqual(q, coords.linear_to_pegasus(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['pegasus_index'], q)
            self.assertEqual(v, coords.pegasus_to_linear(q))
            self.assertEqual(q, coords.linear_to_pegasus(v))

        self.assertEqual(EG, sorted(map(sorted, coords.iter_pegasus_to_linear_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.iter_linear_to_pegasus_pairs(Gn.edges()))))

    def test_graph_relabeling(self):
        def graph_equal(g, h):
            self.assertEqual(set(g), set(h))
            self.assertEqual(
                set(map(tuple, map(sorted, g.edges))),
                set(map(tuple, map(sorted, g.edges)))
            )
            for v, d in g.nodes(data=True):
                self.assertEqual(h.nodes[v], d)

        coords = dnx.pegasus_coordinates(3)
        nodes_nice = dnx.pegasus_graph(3, nice_coordinates=True)
        nodes_linear = list(coords.iter_nice_to_linear(nodes_nice))
        nodes_pegasus = list(coords.iter_nice_to_pegasus(nodes_nice))
        
        for data in True, False:
            p3l = dnx.pegasus_graph(3, data=data).subgraph(nodes_linear)
            p3p = dnx.pegasus_graph(3, data=data, coordinates=True).subgraph(nodes_pegasus)
            p3n = dnx.pegasus_graph(3, data=data, nice_coordinates=True)

            graph_equal(p3l, coords.graph_to_linear(p3l))
            graph_equal(p3l, coords.graph_to_linear(p3p))
            graph_equal(p3l, coords.graph_to_linear(p3n))
            
            graph_equal(p3p, coords.graph_to_pegasus(p3l))
            graph_equal(p3p, coords.graph_to_pegasus(p3p))
            graph_equal(p3p, coords.graph_to_pegasus(p3n))

            graph_equal(p3n, coords.graph_to_nice(p3l))
            graph_equal(p3n, coords.graph_to_nice(p3p))
            graph_equal(p3n, coords.graph_to_nice(p3n))

        h = dnx.pegasus_graph(2)
        del h.graph['labels']
        with self.assertRaises(ValueError):
            coords.graph_to_nice(h)
        with self.assertRaises(ValueError):
            coords.graph_to_linear(h)
        with self.assertRaises(ValueError):
            coords.graph_to_pegasus(h)

    def test_sublattice_mappings(self):
        def check_subgraph_mapping(f, g, h):
            for v in g:
                if not h.has_node(f(v)):
                    raise RuntimeError(f"node {v} mapped to {f(v)} is not in {h.graph['name']} ({h.graph['labels']})")
            for u, v in g.edges:
                if not h.has_edge(f(u), f(v)):
                    raise RuntimeError(f"edge {(u, v)} mapped to {(f(u), f(v))} not present in {h.graph['name']} ({h.graph['labels']})")

        coords5 = dnx.pegasus_coordinates(5)
        coords3 = dnx.pegasus_coordinates(3)

        p3l = dnx.pegasus_graph(3)
        p3c = dnx.pegasus_graph(3, coordinates=True)
        p3n = coords3.graph_to_nice(p3c)

        p5l = dnx.pegasus_graph(5)
        p5c = dnx.pegasus_graph(5, coordinates=True)
        p5n = coords5.graph_to_nice(p5c)

        for target in p5l, p5c, p5n:
            for source in p3l, p3c, p3n:
                covered = set()
                for f in dnx.pegasus_sublattice_mappings(source, target):
                    check_subgraph_mapping(f, source, target)
                    covered.update(map(f, source))
                self.assertEqual(covered, set(target))

        c2l = dnx.chimera_graph(2)
        c2c = dnx.chimera_graph(2, coordinates=True)
        c23l = dnx.chimera_graph(2, 3)
        c32c = dnx.chimera_graph(3, 2, coordinates=True)

        p5n = dnx.pegasus_graph(5, nice_coordinates=True)
        p5l = coords5.graph_to_linear(p5n)
        p5c = coords5.graph_to_pegasus(p5n)

        for target in p5l, p5c, p5n:
            for source in c2l, c2c, c23l, c32c, target:
                covered = set()
                for f in dnx.pegasus_sublattice_mappings(source, target):
                    check_subgraph_mapping(f, source, target)
                    covered.update(map(f, source))
                self.assertEqual(covered, set(target))

    def test_node_list(self):
        m = 4
        G = dnx.pegasus_graph(m)
        # Valid (default) node_list
        node_list = list(G.nodes)
        G = dnx.pegasus_graph(m, node_list=node_list,
                              check_node_list=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        
        with self.assertRaises(ValueError):
            # Invalid node_list on any shape m pegasus graph
            node_list = [0,m*(m-1)*24]
            G = dnx.pegasus_graph(m, node_list=node_list, fabric_only=False,
                                  check_node_list=True)
        with self.assertRaises(ValueError):
            # Invalid node_list due to duplicates
            node_list = [0, 0]
            G = dnx.pegasus_graph(m, node_list=node_list, fabric_only=False,
                                  check_node_list=True)
        
        with self.assertRaises(ValueError):
            # Invalid node_list (fabric_only)
            node_list = [0]
            G = dnx.pegasus_graph(m, node_list=node_list, fabric_only=True,
                                  check_node_list=True)
        with self.assertRaises(ValueError):
            # Invalid node_list (nice_coordinates)
            node_list = [dnx.pegasus_coordinates(m).linear_to_nice(0)]
            G = dnx.pegasus_graph(m, node_list=node_list, nice_coordinates=True,
                                  check_node_list=True)

            
        # Valid coordinate presentation:
        node_list = [(0,0,0,0)]
        G = dnx.pegasus_graph(m, node_list=node_list, fabric_only=False,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        with self.assertRaises(ValueError):
            # Incompatible coordinate presentation:
            node_list = [0]
            G = dnx.pegasus_graph(m, node_list=node_list, fabric_only=False,
                                  check_node_list=True, coordinates=True)
        # Edges are not checked, but node_list is, the edge is deleted:
        edge_list = [(-1,0)]
        node_list = [0]
        G = dnx.pegasus_graph(m, node_list=node_list, edge_list=edge_list, fabric_only=False,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(G.number_of_nodes(), 1)
        # Edges are not checked, but node_list is, the invalid node (-1) is permitted
        # because it is specified in edge_list:
        edge_list = [(-1,0)]
        node_list = [-1,0]
        G = dnx.pegasus_graph(m, node_list=node_list, edge_list=edge_list, fabric_only=False,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 1)
        self.assertEqual(G.number_of_nodes(), 2)
        
    def test_edge_list(self):
        m = 4
        G = dnx.pegasus_graph(m)
        edge_list = list(G.edges)
        # Valid (default) edge_list
        G = dnx.pegasus_graph(m, edge_list=edge_list,
                              check_edge_list=True)
        self.assertEqual(G.number_of_edges(),len(edge_list))
        # Valid edge_list in coordinate system
        edge_list = [((0, 0, 2, 0), (0, 0, 2, 1))]
        G = dnx.pegasus_graph(m, edge_list=edge_list,
                              check_edge_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), len(edge_list))
        
        # Valid edge, but absent from node_list, hence dropped:
        G = dnx.pegasus_graph(m, fabric_only=False)
        edge_list = [(0,1)]
        node_list = [0,2]
        G = dnx.pegasus_graph(m, edge_list=edge_list, node_list = node_list,
                              fabric_only=False, check_edge_list=True)
        self.assertEqual(G.number_of_edges(),0)
        
        with self.assertRaises(ValueError):
            # Invalid edge_list.
            edge_list = [(0,2)] #Vertical next nearest, no edge.
            G = dnx.pegasus_graph(m, edge_list=edge_list, fabric_only=False,
                                  check_edge_list=True)
                    
        with self.assertRaises(ValueError):
            # Edge list has duplicates
            edge_list = [(0, 1), (0, 1)]
            G = dnx.pegasus_graph(m, edge_list=edge_list, fabric_only=False,
                                  check_edge_list=True)

class TestTupleFragmentation(unittest.TestCase):

    def test_empty_list(self):
        # Set up fragmentation function
        pg = dnx.pegasus_graph(3)
        fragment_tuple = get_tuple_fragmentation_fn(pg)

        # Fragment pegasus coordinates
        fragments = fragment_tuple([])
        self.assertEqual([], fragments)

    def test_single_horizontal_coordinate(self):
        # Set up fragmentation function
        pg = dnx.pegasus_graph(2)
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
        pg = dnx.pegasus_graph(6)
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
        pg = dnx.pegasus_graph(6)
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
        pg = dnx.pegasus_graph(2)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = []
        pegasus_coords = defragment_tuple(chimera_coords)

        self.assertEqual([], pegasus_coords)

    def test_single_fragment(self):
        # Set up defragmentation function
        pg = dnx.pegasus_graph(4)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(3, 7, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(0, 1, 2, 0)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_multiple_fragments_from_same_qubit(self):
        # Set up defragmentation function
        pg = dnx.pegasus_graph(3)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(9, 8, 1, 1), (9, 11, 1, 1)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = [(1, 1, 7, 1)]
        self.assertEqual(expected_pegasus_coords, pegasus_coords)

    def test_mixed_fragments(self):
        # Set up defragmenation function
        pg = dnx.pegasus_graph(8)
        defragment_tuple = get_tuple_defragmentation_fn(pg)

        # De-fragment chimera coordinates
        chimera_coords = [(17, 14, 0, 0), (22, 14, 0, 0), (24, 32, 1, 0), (1, 31, 0, 0)]
        pegasus_coords = defragment_tuple(chimera_coords)

        expected_pegasus_coords = {(0, 2, 4, 2), (1, 4, 0, 4), (0, 5, 2, 0)}
        self.assertEqual(expected_pegasus_coords, set(pegasus_coords))

class TestFragmentedEdges(unittest.TestCase):
    def test_linear_indices(self):
        p = dnx.pegasus_graph(3, coordinates=False)
        c = dnx.chimera_graph(24, coordinates=True)
        num_edges = 0
        for u, v in fragmented_edges(p):
            self.assertTrue(c.has_edge(u, v))
            num_edges += 1
        #This is a weird edgecount: each node produces 5 extra edges for the internal connections
        #between fragments corresponding to a pegasus qubit.  But then we need to delete the odd
        #couplers, which aren't included in the chimera graph -- odd couplers make a perfect
        #matching, so thats 1/2 an edge per node.
        self.assertEqual(p.number_of_edges() + 9 * p.number_of_nodes()//2, num_edges)

    def test_coordinates(self):
        p = dnx.pegasus_graph(3, coordinates=True)
        c = dnx.chimera_graph(24, coordinates=True)
        num_edges = 0
        for u, v in fragmented_edges(p):
            self.assertTrue(c.has_edge(u, v))
            num_edges += 1

        #This is a weird edgecount: each node produces 5 extra edges for the internal connections
        #between fragments corresponding to a pegasus qubit.  But then we need to delete the odd
        #couplers, which aren't included in the chimera graph -- odd couplers make a perfect
        #matching, so thats 1/2 an edge per node.
        self.assertEqual(p.number_of_edges() + 9 * p.number_of_nodes()//2, num_edges)

    def test_nice_coordinates(self):
        p = dnx.pegasus_graph(3, nice_coordinates=True)
        c = dnx.chimera_graph(24, coordinates=True)
        num_edges = 0
        for u, v in fragmented_edges(p):
            self.assertTrue(c.has_edge(u, v))
            num_edges += 1
        #This is a weird edgecount: each node produces 5 extra edges for the internal connections
        #between fragments corresponding to a pegasus qubit.  But then we need to delete the odd
        #couplers, which aren't included in the chimera graph -- odd couplers make a perfect
        #matching, so thats 1/2 an edge per node.
        self.assertEqual(p.number_of_edges() + 9 * p.number_of_nodes()//2, num_edges)

    def tests_list(self):
        #Test correct handling of nodes and edges:
        m=4
        num_var = (m-1)*(m-1)*24
        to_coord = dnx.pegasus_coordinates(m).linear_to_pegasus
        node_list_lin = [0, 2, -1]
        node_list = [to_coord(i) for i in node_list_lin]
        G = dnx.pegasus_torus(m=m, node_list = [node_list[i] for i in range(2) ])
        self.assertEqual(G.number_of_edges(),1)
        self.assertEqual(G.number_of_nodes(),2)
        with self.assertRaises(ValueError):
            #Invalid node
            G = dnx.pegasus_torus(m=m, node_list=node_list)
        edge_list_lin = [(0, 1), (m-1, m), (2, 3)] 
        edge_list = [(to_coord(n1), to_coord(n2)) for n1, n2 in edge_list_lin]
        
        G = dnx.pegasus_torus(m=m, edge_list = [edge_list[i] for i in range(2) ])
        
        self.assertEqual(G.number_of_edges(),2)
        self.assertEqual(G.number_of_nodes(),num_var) #No deletions
        
        with self.assertRaises(ValueError):
            #2 invalid, 1 valid
            G = dnx.pegasus_torus(m=m, edge_list = edge_list)
