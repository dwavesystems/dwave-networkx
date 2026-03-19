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

import dwave.graphs as dnx

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = False

try:
    import numpy as np
except ImportError:
    np = False

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
