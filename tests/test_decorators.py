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

import dwave_networkx as dnx
from dwave_networkx.utils import binary_quadratic_model_sampler


class MockSampler:
    def sample_ising(self, h, J):
        pass

    def sample_qubo(self, Q):
        pass


@binary_quadratic_model_sampler(1)
def mock_function(G, sampler=None, **sampler_args):
    assert sampler is not None


class TestDecorators(unittest.TestCase):

    def test_default_set(self):
        dnx.set_default_sampler(MockSampler())
        mock_function(0)
        dnx.unset_default_sampler()

        self.assertEqual(dnx.get_default_sampler(), None,
                         "sampler did not unset correctly")

    def test_no_sampler_set(self):
        with self.assertRaises(dnx.DWaveNetworkXMissingSampler):
            mock_function(0)

    def test_sampler_provided(self):
        mock_function(0, MockSampler())
