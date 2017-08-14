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

        self.assertEqual(dnx.get_default_sampler(), None, "sampler did not unset correctly")

    def test_no_sampler_set(self):
        with self.assertRaises(dnx.DWaveNetworkXMissingSampler):
            mock_function(0)

    def test_sampler_provided(self):
        mock_function(0, MockSampler())
