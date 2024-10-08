# Copyright 2024 D-Wave Systems Inc.
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

import dwave_networkx as dnx
import numpy as np

class TestRegularColoring(unittest.TestCase):

    def test_valid(self):
        test_cases = {'chimera': [(3,3), (4,4)],
                      'pegasus': [(2,), (4,)],
                      'zephyr': [(1,2), (3,4)]}

        for topology_type, topology_shapes in test_cases.items():
            if topology_type == 'zephyr':
                graph = dnx.zephyr_graph
                color = dnx.zephyr_four_color
                num_colors = 4
            elif topology_type == 'pegasus':
                graph = dnx.pegasus_graph
                color = dnx.pegasus_four_color
                num_colors = 4
            elif topology_type == 'chimera':
                graph = dnx.chimera_graph
                color = dnx.chimera_two_color
                num_colors = 2
            else:
                raise ValueError('unrecognized topology')
            
            for topology_shape in topology_shapes:
                G = graph(*topology_shape, coordinates=True)
                col_dict = {q: color(q) for q in G.nodes}
                self.assertTrue(np.all(np.unique(list(col_dict.values())) ==
                                       np.arange(num_colors)))
                self.assertTrue(all(col_dict[q1] != col_dict[q2]
                                    for q1, q2 in G.edges),
                                f'{topology_type}[{topology_shape}]')

    def test_non_default_schemes(self):
        graph = dnx.zephyr_graph
        color = dnx.zephyr_four_color
        num_colors = 4
        topology_type = 'zephyr'
        topology_shapes =  [(2,2), (3,1)]
        for topology_shape in topology_shapes:
            G = graph(*topology_shape, coordinates=True)
            col_dicts = {}
            for scheme in [0,1]:
                col_dict = {q: color(q, scheme=scheme) for q in G.nodes}
                self.assertTrue(np.all(np.unique(list(col_dict.values())) ==
                                       np.arange(num_colors)))
                self.assertTrue(all(col_dict[q1] != col_dict[q2]
                                    for q1, q2 in G.edges),
                                f'{topology_type}[{topology_shape}]')
                col_dicts[scheme] = col_dict
            self.assertFalse(all(col_dicts[0][q] == col_dicts[1][q] for q in G.nodes))
