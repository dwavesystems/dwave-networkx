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
import os
import unittest 
from itertools import product

import networkx as nx
import numpy as np

from dwave_networkx.drawing.visualize_parallel_embeddings import generate_edge_color_dict, generate_node_color_dict, visualize_parallel_embeddings
from dwave_networkx import chimera_graph


_display = os.environ.get("DISPLAY", "") != ""

class TestEmbeddings(unittest.TestCase):

    @unittest.skipUnless(_display, " No display found")
    def test_visualize_embeddings(self):
        S = chimera_graph(1)
        T = chimera_graph(2)
        embeddings = [{i: i+8*offset for i in range(8)} for offset in range(4)]
        visualize_parallel_embeddings(G=T, embeddings=embeddings, S=S)
    
    def test_color_dict(self):
        # The refactored code defines these helpers:
        # generate_node_color_dict(G, embeddings, S=None, one_to_iterable=False, shuffle_colormap=True, seed=None)
        # generate_edge_color_dict(G, embeddings, S, one_to_iterable, node_color_dict)

        T = chimera_graph(2)
        embeddings = [{}]

        # Test empty embeddings: All nodes should be NaN, no edges
        n, _emb = generate_node_color_dict(T, embeddings, S=None, one_to_iterable=False, shuffle_colormap=False)
        e = generate_edge_color_dict(T, _emb, S=None, one_to_iterable=False, node_color_dict=n)
        self.assertTrue(np.all(np.isnan(list(n.values()))))
        self.assertEqual(len(e), 0)

        blocks_of = [1, 8]
        one_to_iterable = [True, False]
        for b, o in product(blocks_of, one_to_iterable):
            if o:
                embeddings = [
                    {0: tuple(n + idx * b for n in range(b))}
                    for idx in range(T.number_of_nodes() // b)
                ]
            else:
                embeddings = [
                    {n: n + idx * b for n in range(b)}
                    for idx in range(T.number_of_nodes() // b)
                ]
            
            # No shuffle
            n, _emb = generate_node_color_dict(T, embeddings, S=None, one_to_iterable=o, shuffle_colormap=False)
            e = generate_edge_color_dict(T, _emb, S=None, one_to_iterable=o, node_color_dict=n)
            # length of e should match certain pattern
            self.assertEqual(len(e), len(embeddings) * (b // 2) * (b // 2))
            
            # With shuffle and seed
            n1, _emb1 = generate_node_color_dict(T, embeddings, S=None, one_to_iterable=o, shuffle_colormap=True, seed=42)
            e1 = generate_edge_color_dict(T, _emb1, S=None, one_to_iterable=o, node_color_dict=n1)
            
            if b == 1:
                # Check color ordering behavior
                vals_n1 = np.array(list(n1.values()))
                # Shuffled version (n1) vs non-shuffled (n)
                vals_n = np.array(list(n.values()))
                # checkeordering differences
                # Highly unlikely to have aligned colors
                self.assertFalse(all(vals_n1[i] <= vals_n1[i+1] for i in range(len(vals_n1)-1)))
                vals_n = np.array([n[v] for v in sorted(n.keys())])
                self.assertTrue(all(vals_n[i] <= vals_n[i+1] for i in range(len(vals_n)-1)))


            # Re-generate with same seed
            n2, _emb2 = generate_node_color_dict(T, embeddings, S=None, one_to_iterable=o, shuffle_colormap=True, seed=42)
            e2 = generate_edge_color_dict(T, _emb2, S=None, one_to_iterable=o, node_color_dict=n2)
            self.assertEqual(n1, n2)
            self.assertEqual(e1, e2)

        S = nx.Graph()
        S.add_node(0)
        embs = [{0: n} for n in T.nodes]

        # With S specified but no edges in S
        n, _emb = generate_node_color_dict(T, embs, S=S, one_to_iterable=False, shuffle_colormap=False)
        e = generate_edge_color_dict(T, _emb, S=S, one_to_iterable=False, node_color_dict=n)
        vals = np.array(list(n.values()))
        self.assertTrue(np.all(np.logical_and(vals < len(embs), vals >= 0)))
        self.assertEqual(len(e), 0)

        # Add edges to S
        S.add_edges_from(list(T.edges)[:2])
        emb = {n: n for n in T.nodes}
        n, _emb = generate_node_color_dict(T, embeddings=[emb], S=S, one_to_iterable=False, shuffle_colormap=False)
        e = generate_edge_color_dict(T, _emb, S=S, one_to_iterable=False, node_color_dict=n)
        vals = np.array(list(n.values()))
        self.assertEqual(len(vals), T.number_of_nodes())
        # 3 nodes should be colored, rest NaN
        self.assertEqual(np.sum(np.isnan(vals)), T.number_of_nodes() - 3)
        self.assertEqual(len(e), 2)

        # Without S, all nodes/edges colored
        n, _emb = generate_node_color_dict(T, embeddings=[emb], S=None, one_to_iterable=False, shuffle_colormap=False)
        e = generate_edge_color_dict(T, _emb, S=None, one_to_iterable=False, node_color_dict=n)
        self.assertEqual(len(e), T.number_of_edges())
        self.assertEqual(len(n), T.number_of_nodes())

        S = nx.from_edgelist({(i, i + 1) for i in range(2)})
        embedding = {i: (i, i + 4) for i in range(3)}
        embeddings = [
            embedding,
            {k: tuple(v + 8 for v in c) for k, c in embedding.items()},
        ]
        # Test one_to_iterable with S
        n, _emb = generate_node_color_dict(T, embeddings, S=S, one_to_iterable=True, shuffle_colormap=False)
        e = generate_edge_color_dict(T, _emb, S=S, one_to_iterable=True, node_color_dict=n)
        vals = np.array(list(n.values()))
        self.assertEqual(len(e), 7 * 2)  # Matches original test logic
        self.assertEqual(len(vals), T.number_of_nodes())
        self.assertEqual(np.sum(np.isnan(vals)), T.number_of_nodes() - 6 * 2)