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

import dwave_networkx as dnx

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

try:
    import numpy as np
except ImportError:
    np = False

# from dwave_networkx.drawing.qubit_layout import generate_edge_color_dict, generate_node_color_dict
# from itertools import product
# import networkx as nx 

_display = os.environ.get('DISPLAY', '') != ''


@unittest.skipUnless(np and plt, "No numpy or matplotlib")
class TestDrawing(unittest.TestCase):
    @unittest.skipUnless(_display, " No display found")
    def test_draw_qubit_graph_kwargs(self):
        G = dnx.chimera_graph(2, 2, 4)
        pos = dnx.chimera_layout(G)
        linear_biases = {v: -v / max(G) if v % 2 == 0 else v / max(G) for v in G}
        quadratic_biases = {(u, v): (u - v) / max(abs(u), abs(v)) for u, v in G.edges}
        cm = plt.get_cmap("spring_r")

        # Don't supply biases
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos)

        # Supply both biases
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, quadratic_biases)

        # Supply linear but not quadratic biases
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, None)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, None, cmap=None)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, None, cmap=cm)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, None, vmin=-0.1, vmax=0)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, linear_biases, None, vmin=0.0, vmax=10)

        # Supply quadratic but not linear biases
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, {}, quadratic_biases)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, None, quadratic_biases)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, None, quadratic_biases, edge_cmap=None)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, None, quadratic_biases, edge_cmap=cm)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, None, quadratic_biases,
                                                  edge_vmin=-0.1, edge_vmax=0)
        dnx.drawing.qubit_layout.draw_qubit_graph(G, pos, None, quadratic_biases,
                                                  edge_vmin=0.0, edge_vmax=10)


    # def test_visualize_parallel_embeddings(self):
    #     # NB, matplotlib.pyplot.show() can be added to display plots
    #     import matplotlib.pyplot as plt

    #     embeddings = [{}]
    #     T = dnx.chimera_graph(2)
    #     n = generate_node_color_dict(T, embeddings)
    #     e = generate_edge_color_dict(T, embeddings, node_color_dict=n)
    #     self.assertTrue(np.all(np.isnan(list(n.values()))))
    #     self.assertEqual(len(e), 0)
    #     blocks_of = [1, 8]
    #     one_to_iterable = [True, False]
    #     for b, o in product(blocks_of, one_to_iterable):
    #         if o:
    #             embeddings = [
    #                 {0: tuple(n + idx * b for n in range(b))}
    #                 for idx in range(T.number_of_nodes() // b)
    #             ]  # Blocks of 8
    #         else:
    #             embeddings = [
    #                 {n: n + idx * b for n in range(b)}
    #                 for idx in range(T.number_of_nodes() // b)
    #             ]  # Blocks of 8

    #         n = generate_node_color_dict(T, embeddings, one_to_iterable=o, shuffle_colormap=False)
    #         e = generate_edge_color_dict(T, embeddings, one_to_iterable=o)
    #         #n, e = visualize_embeddings(
    #         #   T, embeddings, one_to_iterable=o, shuffle_colormap=False
    #         #)
    #         self.assertEqual(len(e), len(embeddings) * (b // 2) * (b // 2))

    #         n1 = generate_node_color_dict(T, embeddings, shuffle_colormap=True, seed=42, one_to_iterable=o)
    #         e1 = generate_edge_color_dict(T, embeddings, one_to_iterable=o, node_color_dict=n1)
    #         # n1, e1 = visualize_embeddings(
    #         #     T, embeddings, shuffle_colormap=True, seed=42, one_to_iterable=o
    #         # )
    #         if b == 1:
    #             # Highly unlikely to have aligned colors.
    #             self.assertFalse(
    #                 all([n1[idx] <= n1[idx + 1] for idx in range(len(n1) - 1)])
    #             )
    #             self.assertTrue(
    #                 all([n[idx] <= n[idx + 1] for idx in range(len(n) - 1)])
    #             )
    #         n2 = generate_node_color_dict(T, embeddings, shuffle_colormap=True, seed=42, one_to_iterable=o)
    #         e2 = generate_edge_color_dict(T, embeddings, one_to_iterable=o, node_color_dict=n1)
    #         # n2, e2 = visualize_embeddings(
    #         #     T, embeddings, shuffle_colormap=True, seed=42, one_to_iterable=o
    #         # )
    #         self.assertEqual(n1, n2)
    #         self.assertEqual(e1, e2)

    #     S = nx.Graph()
    #     S.add_node(0)
        # embs = [{0: n} for n in T.nodes]
        # n, e = visualize_embeddings(
        #     T, embeddings=embs, S=S
        # )  # Should plot every node colorfully but no edges
        # vals = np.array(list(n.values()))
        # self.assertTrue(np.all(np.logical_and(vals < len(embs), vals >= 0)))
        # self.assertEqual(len(e), 0)

        # S.add_edges_from(list(T.edges)[:2])
        # emb = {n: n for n in T.nodes}

        # n, e = visualize_embeddings(
        #     T, embeddings=[emb], S=S
        # )  # Should plot 3 nodes, and two edges

        # vals = np.array(list(n.values()))
        # self.assertEqual(len(vals), T.number_of_nodes())
        # self.assertEqual(np.sum(np.isnan(vals)), T.number_of_nodes() - 3)
        # self.assertEqual(len(e), 2)

        # n, e = visualize_embeddings(
        #     T, embeddings=[emb], S=None
        # )  # Should plot every nodes and edges
        # self.assertEqual(len(e), T.number_of_edges())
        # self.assertEqual(len(n), T.number_of_nodes())

        # S = nx.from_edgelist({(i, i + 1) for i in range(2)})
        # embedding = {i: (i, i + 4) for i in range(3)}
        # embeddings = [
        #     embedding,
        #     {k: tuple(v + 8 for v in c) for k, c in embedding.items()},
        # ]
        # plt.figure("check one_to_iterable with S")
        # n, e = visualize_embeddings(T, S=S, embeddings=embeddings, one_to_iterable=True)
        # vals = np.array(list(n.values()))
        # self.assertEqual(len(e), 7 * 2)
        # self.assertEqual(len(vals), T.number_of_nodes())
        # self.assertEqual(np.sum(np.isnan(vals)), T.number_of_nodes() - 6 * 2)
        # # Should plot 7 edges total and 6 nodes: Top left and top right cells.
        # #  3 (NE to SW) chains, 2 logical double-couplers
        # # NB without S would be a clique.