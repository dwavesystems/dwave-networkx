# Copyright 2021 D-Wave Systems Inc.
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
import numpy as np

class TestZephyrGraph(unittest.TestCase):
    def test_single_tile(self):

        # fully specified
        G = dnx.zephyr_graph(1, 4)

        # should have 8 nodes
        self.assertEqual(len(G), 48)

        # nodes 0,...,7 should be in the tile
        for n in range(48):
            self.assertIn(n, G)


    def test_not_full_yield(self):
        edges =  [(2, 30), (7, 44), (10, 37), (12, 29), (15, 37), (19, 41)]
        G = dnx.zephyr_graph(1, 4, edge_list=edges)
        for e in edges:
            self.assertIn(e, G.edges())
        for (u, v) in G.edges:
            self.assertTrue((u, v) in edges or (v, u) in edges)

        nodes = [0, 1, 2]
        G = dnx.zephyr_graph(1, 2, node_list=nodes)
        self.assertEqual(len(G), 3)
        self.assertEqual(len(G.edges()), 1)

        edges = [(0, 1), (2, 3)]
        nodes = [0, 1, 2, 3]
        G = dnx.zephyr_graph(1, 2, node_list=nodes, edge_list=edges)
        self.assertEqual(len(G), 4)
        self.assertEqual(len(G.edges()), 2)

    def test_float_robustness(self):
        G = dnx.zephyr_graph(8 / 2)

        self.assertEqual(set(G.nodes), set(dnx.zephyr_graph(4).nodes))
        for u, v in dnx.zephyr_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.zephyr_graph(4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.zephyr_graph(4).nodes))
        for u, v in dnx.zephyr_graph(4).edges:
            self.assertIn(u, G[v])

    def test_coordinate_basics(self):
        from dwave.graphs.generators.zephyr import zephyr_coordinates
        G = dnx.zephyr_graph(4)
        H = dnx.zephyr_graph(4, coordinates=True)
        coords = zephyr_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['zephyr_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['zephyr_index'], q)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))

    def test_coordinate_subgraphs(self):
        from dwave.graphs.generators.zephyr import zephyr_coordinates
        from random import sample
        G = dnx.zephyr_graph(4)
        H = dnx.zephyr_graph(4, coordinates=True)
        coords = zephyr_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.iter_linear_to_zephyr(lmask))

        self.assertEqual(lmask, list(coords.iter_zephyr_to_linear(cmask)))

        Gm = dnx.zephyr_graph(4, node_list=lmask)
        Hm = dnx.zephyr_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.zephyr_graph(4, edge_list=EG)
        Hn = dnx.zephyr_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['zephyr_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['zephyr_index'], q)
            self.assertEqual(v, coords.zephyr_to_linear(q))
            self.assertEqual(q, coords.linear_to_zephyr(v))

        self.assertEqual(EG, sorted(map(sorted, coords.iter_zephyr_to_linear_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.iter_linear_to_zephyr_pairs(Gn.edges()))))

    def test_graph_relabeling(self):
        def graph_equal(g, h):
            self.assertEqual(set(g), set(h))
            self.assertEqual(
                set(map(tuple, map(sorted, g.edges))),
                set(map(tuple, map(sorted, g.edges)))
            )
            for v, d in g.nodes(data=True):
                self.assertEqual(h.nodes[v], d)

        coords = dnx.zephyr_coordinates(3)
        for data in True, False:
            z3l = dnx.zephyr_graph(3, data=data)
            z3c = dnx.zephyr_graph(3, data=data, coordinates=True)

            graph_equal(z3l, coords.graph_to_linear(z3l))
            graph_equal(z3l, coords.graph_to_linear(z3c))
            
            graph_equal(z3c, coords.graph_to_zephyr(z3c))
            graph_equal(z3c, coords.graph_to_zephyr(z3l))

        h = dnx.zephyr_graph(2)
        del h.graph['labels']
        with self.assertRaises(ValueError):
            coords.graph_to_linear(h)
        with self.assertRaises(ValueError):
            coords.graph_to_zephyr(h)


    def test_sublattice_mappings(self):
        def check_subgraph_mapping(f, g, h):
            for v in g:
                if not h.has_node(f(v)):
                    raise RuntimeError(f"node {v} mapped to {f(v)} is not in {h.graph['name']} ({h.graph['labels']})")
            for u, v in g.edges:
                if not h.has_edge(f(u), f(v)):
                    raise RuntimeError(f"edge {(u, v)} mapped to {(f(u), f(v))} not present in {h.graph['name']} ({h.graph['labels']})")

        z2l = dnx.zephyr_graph(2)
        z2c = dnx.zephyr_graph(2, coordinates=True)
        c2l = dnx.chimera_graph(2)
        c2c = dnx.chimera_graph(2, coordinates=True)
        c23l = dnx.chimera_graph(2, 3)
        c32c = dnx.chimera_graph(3, 2, coordinates=True)
        c2l8 = dnx.chimera_graph(2, t=8)
        c2c8 = dnx.chimera_graph(2, t=8, coordinates=True)        
        c23l8 = dnx.chimera_graph(2, 3, t=8)
        c32c8 = dnx.chimera_graph(3, 2, t=8, coordinates=True)        

        z5l = dnx.zephyr_graph(5)
        z5c = dnx.zephyr_graph(5, coordinates=True)

        for target in z5l, z5c:
            for source in z2l, z2c, c2l, c2c, c2l8, c2c8, c23l, c32c, c23l8, c32c8, target:
                covered = set()
                for f in dnx.zephyr_sublattice_mappings(source, target):
                    check_subgraph_mapping(f, source, target)
                    covered.update(map(f, source))
                self.assertEqual(covered, set(target))


    def test_node_list(self):
        m=4
        t=2
        N = 4 * t * m * (2 * m + 1)
        G = dnx.chimera_graph(m,t)
        # Valid (full) node_list
        node_list = list(G.nodes)
        G = dnx.zephyr_graph(m, t, node_list=node_list,
                              check_node_list=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        # Valid node_list in coordinate system
        node_list = [(0, 0, 0, 0, 0)]
        G = dnx.zephyr_graph(m, t, node_list=node_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        with self.assertRaises(ValueError):
            # Invalid node
            node_list = [0, N]
            G = dnx.zephyr_graph(m, t, node_list=node_list,
                                  check_node_list=True)
        with self.assertRaises(ValueError):
            # Duplicates
            node_list = [0, 0]
            G = dnx.zephyr_graph(m, node_list=node_list,
                                 check_node_list=True)
    
        with self.assertRaises(ValueError):
            # Not in the requested coordinate system
            node_list = [0]
            G = dnx.zephyr_graph(m, t, node_list=node_list,
                                 check_node_list=True, coordinates=True)
        # Edges are not checked, but node_list is, the edge is deleted:
        edge_list = [(-1,0)]
        node_list = [0]
        G = dnx.zephyr_graph(m, t, node_list=node_list, edge_list=edge_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(G.number_of_nodes(), 1)
        # Edges are not checked, but node_list is, the invalid node (-1) is permitted
        # because it is specified in edge_list:
        edge_list = [(-1,0)]
        node_list = [-1,0]
        G = dnx.zephyr_graph(m, t, node_list=node_list, edge_list=edge_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 1)
        self.assertEqual(G.number_of_nodes(), 2)

    def test_edge_list(self):
        m=2
        t=4
        N = 4 * t * m * (2 * m + 1)
        G = dnx.zephyr_graph(m, t)
        edge_list = list(G.edges)
        # Valid (full) edge_list
        G = dnx.zephyr_graph(m, t, edge_list=edge_list,
                              check_edge_list=True)
        self.assertEqual(G.number_of_edges(),len(edge_list))

        # Valid edge_list in coordinate system
        edge_list = [((0, 0, 0, 0, 0), (0, 0, 0, 0, 1))]
        G = dnx.zephyr_graph(m, t, edge_list=edge_list,
                              check_edge_list=True, coordinates=True)
        
        self.assertEqual(G.number_of_edges(),len(edge_list))

        # Valid edge, but absent from node_list, hence dropped:
        edge_list = [(0,1)]
        node_list = [0,2]
        G = dnx.zephyr_graph(m, t, edge_list=edge_list, node_list = node_list,
                              check_edge_list=True)
        self.assertEqual(G.number_of_edges(), 0)
        
        with self.assertRaises(ValueError):
            # Invalid edge_list (0,N-1).
            edge_list = [(0, N-1), (0, 1)]
            G = dnx.zephyr_graph(m, t, edge_list=edge_list,
                                  check_edge_list=True)
        
        with self.assertRaises(ValueError):
            # Edge list has duplicates
            edge_list = [(0, 1), (0, 1)]
            G = dnx.zephyr_graph(m, t, edge_list=edge_list,
                                  check_edge_list=True)

            
class TestZephyrTorus(unittest.TestCase):
    def test(self):
        for m in [2,3,4]:
            for t in [1,4]:
                G = dnx.zephyr_torus(m=m, t=t)
                # Test bulk properties:
                
                num_nodes = (8*t)*m*m
                self.assertEqual(G.number_of_nodes(), num_nodes)
                if m==1:
                    conn = 1 + t*4;
                    self.assertEqual(G.number_of_edges(),(num_nodes*conn)//2)
                elif m==2:
                    conn = 3 + t*4 
                    self.assertEqual(G.number_of_edges(),(num_nodes*conn)//2)
                else:
                    conn = 4 + t*4;
                    self.assertEqual(G.number_of_edges(),(num_nodes*conn)//2)

                    # Check translational invariance (identical edges, identical nodes):
                    # (u,w,k,j,z) -> (u, [w + u (2*dx) + (1-u)*(2*dy)]%(m-1)), k, j, [z + (1-u) dx + u dy]%(m-1))
                    dx = 1 + np.random.randint(m-2)
                    dy = np.random.randint(m-1)
                    relabel = lambda tup: (tup[0],(tup[1] + tup[0]*(2*dx) + (1-tup[0])*(2*dy))%(2*m),tup[2],tup[3],(tup[4] + tup[0]*dy + (1-tup[0])*dx)%m)
                    G_translated = nx.relabel_nodes(G,relabel,copy=True)
                    G.remove_edges_from(G_translated.edges())
                    self.assertEqual(G.number_of_edges(),0) #At t=1, m=2 (n=32), 8 left over edges. 8 edges collapsed to the same place?
                    G.remove_nodes_from(G_translated.nodes())
                    self.assertEqual(G.number_of_nodes(),0)
                    
                    
    def tests_list(self):
        # Test correct handling of nodes and edges:
        m=3
        t=4
        num_var = m*m*t*8
        to_coord = dnx.zephyr_coordinates(m,t).linear_to_zephyr
        node_list_lin = [0, 1, -1]
        node_list = [to_coord(i) for i in node_list_lin]
        G = dnx.zephyr_torus(m=m, t=t, node_list = [node_list[i] for i in range(2) ])
        self.assertEqual(G.number_of_edges(),1)
        self.assertEqual(G.number_of_nodes(),2)
        with self.assertRaises(ValueError):
            # 1 invalid node
            G = dnx.zephyr_torus(m=m, t=t, node_list=node_list)
        edge_list_lin = [(0, 1), (m, m+1), (0, m+1)]
        edge_list = [(to_coord(n1), to_coord(n2)) for n1, n2 in edge_list_lin]
        G = dnx.zephyr_torus(m=m, t=t, edge_list = [edge_list[i] for i in range(2) ])
        
        self.assertEqual(G.number_of_edges(),2)
        self.assertEqual(G.number_of_nodes(),num_var) # No deletions
        
        with self.assertRaises(ValueError):
            # 1 invalid edge
            G = dnx.zephyr_torus(m=m, t=t, edge_list = edge_list)
            
