# Copyright 2019 D-Wave Systems Inc.
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

import dwave.graphs as dnx


class Test_markov_network(unittest.TestCase):
    def test_empty(self):
        MN = dnx.markov_network({})

    def test_one_node(self):
        MN = dnx.markov_network({'a': {(0,): 1.2, (1,): .4}})

    def test_one_edge(self):
        MN = dnx.markov_network({'ab': {(0, 0): 1.2, (1, 0): .4, (0, 1): 1.3, (1, 1): -4}})
