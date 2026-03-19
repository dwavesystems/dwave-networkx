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
import numpy as np

alpha_map = dict(enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))


class TestChimeraGraph(unittest.TestCase):
    def test_single_tile(self):

        # fully specified
        G = dnx.chimera_graph(1, 1, 4)

        # should have 8 nodes
        self.assertEqual(len(G), 8)

        # nodes 0,...,7 should be in the tile
        for n in range(8):
            self.assertIn(n, G)

        # check bipartite
        for i in range(4):
            for j in range(4, 8):
                self.assertTrue((i, j) in G.edges() or (j, i) in G.edges())

    def test_find_chimera_indices_single_tile(self):
        for k in range(1, 10):
            G = dnx.chimera_graph(1, 1, k)

            # get the chimera indices, check that they match the ones already
            # present
            chimera_indices = dnx.find_chimera_indices(G)
            self._check_matching_chimera_indices(G, chimera_indices)

    def test_find_chimera_indices_single_tile_alpha_labels(self):
        for k in range(1, 10):
            G = nx.relabel_nodes(dnx.chimera_graph(1, 1, k), alpha_map)

            # get the chimera indices, check that they match the ones already
            # present
            chimera_indices = dnx.find_chimera_indices(G)
            self._check_matching_chimera_indices(G, chimera_indices)

    def test_find_chimera_indices_one_tile_degenerate(self):
        G = dnx.chimera_graph(1, 1, 5)

        # remove 1 node
        G.remove_node(4)
        chimera_indices = dnx.find_chimera_indices(G)
        self._check_matching_chimera_indices(G, chimera_indices)

        # remove another node
        G.remove_node(3)
        chimera_indices = dnx.find_chimera_indices(G)
        self._check_matching_chimera_indices(G, chimera_indices)

    def test_not_full_yield(self):
        edges = [(0, 3), (0, 2), (1, 3)]
        G = dnx.chimera_graph(1, 1, 2, edge_list=edges)
        for e in edges:
            self.assertIn(e, G.edges())
        for (u, v) in G.edges:
            self.assertTrue((u, v) in edges or (v, u) in edges)

        nodes = [0, 1, 2]
        G = dnx.chimera_graph(1, 1, 2, node_list=nodes)
        self.assertEqual(len(G), 3)
        self.assertEqual(len(G.edges()), 2)

        edges = [(0, 2), (1, 2)]
        nodes = [0, 1, 2, 3]
        G = dnx.chimera_graph(1, 1, 2, node_list=nodes, edge_list=edges)
        # 3 should be added as a singleton
        self.assertEqual(len(G[3]), 0)

    def test_float_robustness(self):
        G = dnx.chimera_graph(8 / 2)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.chimera_graph(4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

        G = dnx.chimera_graph(4, 4, 4.)

        self.assertEqual(set(G.nodes), set(dnx.chimera_graph(4).nodes))
        for u, v in dnx.chimera_graph(4).edges:
            self.assertIn(u, G[v])

    # def test_find_chimera_indices_typical(self):
    #     for t in range(2, 5):
    #         G = dnx.chimera_graph(2, 2, t)
    #         chimera_indices = dnx.find_chimera_indices(G)
    #         self._check_matching_chimera_indices(G, chimera_indices)

    #         G = dnx.chimera_graph(4, 4, t)
    #         chimera_indices = dnx.find_chimera_indices(G)
    #         self._check_matching_chimera_indices(G, chimera_indices)

    # def test_find_chimera_indices_shore_1(self):
    #     G = dnx.chimera_graph(2, 2, 1)
    #     chimera_indices = dnx.find_chimera_indices(G)
    #     self._check_matching_chimera_indices(G, chimera_indices)

    #     G = dnx.chimera_graph(4, 4, 1)
    #     chimera_indices = dnx.find_chimera_indices(G)
    #     self._check_matching_chimera_indices(G, chimera_indices)

    def _check_matching_chimera_indices(self, G, chimera_indices):
        for v, dat in G.nodes(data=True):
            self.assertEqual(dat['chimera_index'], chimera_indices[v])

    def test_coordinate_basics(self):
        from dwave.graphs.generators.chimera import chimera_coordinates
        G = dnx.chimera_graph(4)
        H = dnx.chimera_graph(4, coordinates=True)
        coords = chimera_coordinates(4)
        Gnodes = G.nodes
        Hnodes = H.nodes
        for v in Gnodes:
            q = Gnodes[v]['chimera_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.chimera_to_linear(q))
            self.assertEqual(q, coords.linear_to_chimera(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['chimera_index'], q)
            self.assertEqual(v, coords.chimera_to_linear(q))
            self.assertEqual(q, coords.linear_to_chimera(v))

    def test_coordinate_subgraphs(self):
        from dwave.graphs.generators.chimera import chimera_coordinates
        from random import sample
        G = dnx.chimera_graph(4)
        H = dnx.chimera_graph(4, coordinates=True)
        coords = chimera_coordinates(4)

        lmask = sample(list(G.nodes()), G.number_of_nodes()//2)
        cmask = list(coords.iter_linear_to_chimera(lmask))

        self.assertEqual(lmask, list(coords.iter_chimera_to_linear(cmask)))

        Gm = dnx.chimera_graph(4, node_list=lmask)
        Hm = dnx.chimera_graph(4, node_list=cmask, coordinates=True)

        Gs = G.subgraph(lmask)
        Hs = H.subgraph(cmask)

        EG = sorted(map(sorted, Gs.edges()))
        EH = sorted(map(sorted, Hs.edges()))

        self.assertEqual(EG, sorted(map(sorted, Gm.edges())))
        self.assertEqual(EH, sorted(map(sorted, Hm.edges())))

        Gn = dnx.chimera_graph(4, edge_list=EG)
        Hn = dnx.chimera_graph(4, edge_list=EH, coordinates=True)

        Gnodes = Gn.nodes
        Hnodes = Hn.nodes
        for v in Gnodes:
            q = Gnodes[v]['chimera_index']
            self.assertIn(q, Hnodes)
            self.assertEqual(Hnodes[q]['linear_index'], v)
            self.assertEqual(v, coords.chimera_to_linear(q))
            self.assertEqual(q, coords.linear_to_chimera(v))
        for q in Hnodes:
            v = Hnodes[q]['linear_index']
            self.assertIn(v, Gnodes)
            self.assertEqual(Gnodes[v]['chimera_index'], q)
            self.assertEqual(v, coords.chimera_to_linear(q))
            self.assertEqual(q, coords.linear_to_chimera(v))

        self.assertEqual(EG, sorted(map(sorted, coords.iter_chimera_to_linear_pairs(Hn.edges()))))
        self.assertEqual(EH, sorted(map(sorted, coords.iter_linear_to_chimera_pairs(Gn.edges()))))

    def test_linear_to_chimera(self):
        G = dnx.linear_to_chimera(212, 8, 8, 4)
        self.assertEqual(G, (3, 2, 1, 0))

    def test_chimera_to_linear(self):
        G = dnx.chimera_to_linear(3, 2, 1, 0, 8, 8, 4)
        self.assertEqual(G, 212)

    def test_nonsquare_coordinate_generator(self):
        #issue 149 found an issue with non-square generators -- let's be extra careful here
        for (m, n) in [(2, 4), (4, 2)]:
            G = dnx.chimera_graph(m, n, coordinates=True, data=True)
            H = dnx.chimera_graph(m, n, coordinates=False, data=True)
            self.assertTrue(nx.is_isomorphic(G, H))

            Gnodes = set(G.nodes)
            Glabels = set(q['linear_index'] for q in G.nodes.values())

            Hnodes = set(H.nodes)
            Hlabels = set(q['chimera_index'] for q in H.nodes.values())

            self.assertEqual(Gnodes, Hlabels)
            self.assertEqual(Hnodes, Glabels)

            coords = dnx.chimera_coordinates(m, n)
            F = nx.relabel_nodes(G, coords.chimera_to_linear, copy=True)
            self.assertEqual(set(map(frozenset, F.edges)), set(map(frozenset, H.edges)))

            E = nx.relabel_nodes(H, coords.linear_to_chimera, copy=True)
            self.assertEqual(set(map(frozenset, E.edges)), set(map(frozenset, G.edges)))


    def test_graph_relabeling(self):
        def graph_equal(g, h):
            self.assertEqual(set(g), set(h))
            self.assertEqual(
                set(map(tuple, map(sorted, g.edges))),
                set(map(tuple, map(sorted, g.edges)))
            )
            for v, d in g.nodes(data=True):
                self.assertEqual(h.nodes[v], d)

        coords = dnx.chimera_coordinates(3)
        for data in True, False:
            c3l = dnx.chimera_graph(3, data=data)
            c3c = dnx.chimera_graph(3, data=data, coordinates=True)

            graph_equal(c3l, coords.graph_to_linear(c3c))
            graph_equal(c3l, coords.graph_to_linear(c3l))
            graph_equal(c3c, coords.graph_to_chimera(c3l))
            graph_equal(c3c, coords.graph_to_chimera(c3c))

        h = dnx.chimera_graph(2)
        del h.graph['labels']
        with self.assertRaises(ValueError):
            coords.graph_to_linear(h)
        with self.assertRaises(ValueError):
            coords.graph_to_chimera(h)


    def test_sublattice_mappings(self):
        def check_subgraph_mapping(f, g, h):
            for v in g:
                if not h.has_node(f(v)):
                    raise RuntimeError(f"node {v} mapped to {f(v)} is not in {h.graph['name']} ({h.graph['labels']})")
            for u, v in g.edges:
                if not h.has_edge(f(u), f(v)):
                    raise RuntimeError(f"edge {(u, v)} mapped to {(f(u), f(v))} not present in {h.graph['name']} ({h.graph['labels']})")

        c2l = dnx.chimera_graph(2)
        c2c = dnx.chimera_graph(2, coordinates=True)
        c32l = dnx.chimera_graph(3, 2)
        c23c = dnx.chimera_graph(2, 3, coordinates=True)

        c5l = dnx.chimera_graph(5)
        c5c = dnx.chimera_graph(5, coordinates=True)
        c54l = dnx.chimera_graph(5, 4)
        c45c = dnx.chimera_graph(4, 5, coordinates=True)

        for target in c5l, c5c, c54l, c45c:
            for source in c2l, c2c, c32l, c23c, target:
                covered = set()
                for f in dnx.chimera_sublattice_mappings(source, target):
                    check_subgraph_mapping(f, source, target)
                    covered.update(map(f, source))
                self.assertEqual(covered, set(target))


    def test_node_list(self):
        m = 4
        n = 3
        t = 2
        N = m*n*t*2
        G = dnx.chimera_graph(m,n,t)
        # Valid (full) node_list
        node_list = list(G.nodes)
        G = dnx.chimera_graph(m, n, t, node_list=node_list,
                              check_node_list=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        # Valid node_list in coordinate system
        node_list = [(0,0,0,0)]
        G = dnx.chimera_graph(m, n, t, node_list=node_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_nodes(), len(node_list))
        with self.assertRaises(ValueError):
            # Invalid node_list
            node_list = [0, N]
            G = dnx.chimera_graph(m, n, t, node_list=node_list,
                                  check_node_list=True)
        with self.assertRaises(ValueError):
            # Invalid node_list due to duplicates
            node_list = [0, 0]
            G = dnx.chimera_graph(m, node_list=node_list,
                                  check_node_list=True)
        
        with self.assertRaises(ValueError):
            # Node is valid, but not in the requested coordinate system
            node_list = [0]
            G = dnx.chimera_graph(m, n, t, node_list=node_list,
                                  check_node_list=True, coordinates=True)
    
        edge_list = [(-1,0)]
        node_list = [0]
        # Edges are not checked, but node_list is, the edge is deleted:
        G = dnx.chimera_graph(m, n, t, node_list=node_list, edge_list=edge_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 0)
        self.assertEqual(G.number_of_nodes(), 1)
        edge_list = [(-1,0)]
        node_list = [-1,0]
        # Edges are not checked, but node_list is, the invalid node (-1) is permitted
        # because it is specified in edge_list:
        G = dnx.chimera_graph(m, n, t, node_list=node_list, edge_list=edge_list,
                              check_node_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(), 1)
        self.assertEqual(G.number_of_nodes(), 2)
        
            
    def test_edge_list(self):
        m = 2
        n = 3
        t = 4
        num_var = m*n*t*2
        G = dnx.chimera_graph(m, n, t)
        edge_list = list(G.edges)
        # Valid (full) edge_list
        G = dnx.chimera_graph(m, n, t, edge_list=edge_list,
                              check_edge_list=True)
        self.assertEqual(G.number_of_edges(),len(edge_list))
        # Valid edge_list in coordinate system
        edge_list = [((0,0,0,0),(0,0,1,0))]
        G = dnx.chimera_graph(m, n, t, edge_list=edge_list,
                              check_edge_list=True, coordinates=True)
        self.assertEqual(G.number_of_edges(),len(edge_list))
        self.assertEqual(G.number_of_nodes(),num_var) #No node deletions specified
        
        # Valid edge, but absent from node_list, hence dropped:
        edge_list = [(0,t)]
        node_list = list(range(t))
        G = dnx.chimera_graph(m, n, t, edge_list=edge_list, node_list = node_list,
                              check_edge_list=True)
        self.assertEqual(G.number_of_edges(),0)
        
        with self.assertRaises(ValueError):
            # Invalid edge_list (0,1) is a vertical-vertical coupler.
            edge_list = [(0,t),(0,1)]
            G = dnx.chimera_graph(m, n, t, edge_list=edge_list,
                                  check_edge_list=True)
            
        with self.assertRaises(ValueError):
            # Edge list has duplicates
            edge_list = [(0, t), (0, t)]
            G = dnx.chimera_graph(m, edge_list=edge_list,
                                  check_edge_list=True)
       
class TestChimeraTorus(unittest.TestCase):
    def test(self):
        for m in range(1,4):
            for n in range(1,4):
                for t in [1,4]:
                    conn_vert = min(2,m-1) + t
                    conn_horiz = min(2,n-1) + t
                    num_var = m*n*t*2
                    num_edges = ((num_var//2)*(conn_vert + conn_horiz))//2
                    g = dnx.chimera_torus(m=m,n=n,t=t)

                    # Check bulk properties:
                    self.assertEqual(g.number_of_nodes(),num_var) # Number nodes
                    self.assertEqual(g.number_of_edges(),num_edges) # Number nodes
                    
                    # Check translational invariance:
                    if m > 1:
                        drow = 1+np.random.randint(m-1)
                    else:
                        drow = 0
                    if n > 1:
                        dcol = 1+np.random.randint(n-1)
                    else:
                        dcol = 0
                    relabel = lambda tup: ((tup[0]+drow)%m,(tup[1]+dcol)%n,tup[2],tup[3])
                    g_translated = nx.relabel_nodes(g,relabel,copy=True)
                    #Check 1:1 correspondence of edges
                    g.remove_edges_from(g_translated.edges())
                    self.assertEqual(g.number_of_edges(),0)
                    #Check 1:1 correspondence of nodes
                    g.remove_nodes_from(g_translated.nodes())
                    self.assertEqual(g.number_of_nodes(),0)
                    
    def tests_list(self):
        #Test correct handling of nodes and edges:
        m=3
        n=3
        t=2
        num_var = m*n*t*2
        to_coord = dnx.chimera_coordinates(m,n,t).linear_to_chimera
        node_list_lin = [0, t-1, num_var]
        node_list = [to_coord(i) for i in node_list_lin]
        G = dnx.chimera_torus(m=m, n=n, t=t, node_list = [node_list[i] for i in range(2) ])
        self.assertEqual(G.number_of_edges(),0)
        self.assertEqual(G.number_of_nodes(),2)
        with self.assertRaises(ValueError):
            # 1 invalid node
            G = dnx.chimera_torus(m=m, n=n, t=t, node_list=node_list)
        edge_list_lin = [(0, t), (0, t+1), (0, 1)]
        edge_list = [(to_coord(n1), to_coord(n2)) for n1, n2 in edge_list_lin]
        G = dnx.chimera_torus(m=m, n=n, t=t, edge_list = [edge_list[i] for i in range(2) ])
        
        self.assertEqual(G.number_of_edges(),2)
        self.assertEqual(G.number_of_nodes(),num_var) #No deletions
        
        with self.assertRaises(ValueError):
            # 1 invalid edge
            G = dnx.chimera_torus(m=m, n=n, t=t, edge_list = edge_list)
            
