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
import dimod

import dwave.graphs as dnx


class Test_sample_markov_network_bqm(unittest.TestCase):
    def test_one_node(self):
        potentials = {'a': {(0,): 1.2, (1,): .4}}

        bqm = dnx.markov_network_bqm(dnx.markov_network(potentials))

        for edge, potential in potentials.items():
            for config, energy in potential.items():
                sample = dict(zip(edge, config))
                self.assertAlmostEqual(bqm.energy(sample), energy)

    def test_one_edge(self):
        potentials = {'ab': {(0, 0): 1.2, (1, 0): .4,
                             (0, 1): 1.3, (1, 1): -4}}

        bqm = dnx.markov_network_bqm(dnx.markov_network(potentials))

        for edge, potential in potentials.items():
            for config, energy in potential.items():
                sample = dict(zip(edge, config))
                self.assertAlmostEqual(bqm.energy(sample), energy)

    def test_typical(self):
        potentials = {'a': {(0,): 1.5, (1,): -.5},
                      'ab': {(0, 0): 1.2, (1, 0): .4,
                             (0, 1): 1.3, (1, 1): -4},
                      'bc': {(0, 0): 1.7, (1, 0): .4,
                             (0, 1): -1, (1, 1): -4},
                      'd': {(0,): -.5, (1,): 1.6}}

        bqm = dnx.markov_network_bqm(dnx.markov_network(potentials))

        samples = dimod.ExactSolver().sample(bqm)

        for sample, energy in samples.data(['sample', 'energy']):

            en = 0
            for interaction, potential in potentials.items():
                config = tuple(sample[v] for v in interaction)
                en += potential[config]

            self.assertAlmostEqual(en, energy)


class Test_sample_markov_network(unittest.TestCase):
    def test_typical_return_sampleset(self):
        potentials = {'a': {(0,): 1.5, (1,): -.5},
                      'ab': {(0, 0): 1.2, (1, 0): .4,
                             (0, 1): 1.3, (1, 1): -4},
                      'bc': {(0, 0): 1.7, (1, 0): .4,
                             (0, 1): -1, (1, 1): -4},
                      'd': {(0,): -.5, (1,): 1.6}}

        MN = dnx.markov_network(potentials)

        samples = dnx.sample_markov_network(MN, dimod.ExactSolver(),
                                            fixed_variables={'c': 0},
                                            return_sampleset=True)

        for sample, energy in samples.data(['sample', 'energy']):
            self.assertEqual(sample['c'], 0)

            en = 0
            for interaction, potential in potentials.items():
                config = tuple(sample[v] for v in interaction)
                en += potential[config]

            self.assertAlmostEqual(en, energy)

    def test_typical(self):
        potentials = {'a': {(0,): 1.5, (1,): -.5},
                      'ab': {(0, 0): 1.2, (1, 0): .4,
                             (0, 1): 1.3, (1, 1): -4},
                      'bc': {(0, 0): 1.7, (1, 0): .4,
                             (0, 1): -1, (1, 1): -4},
                      'd': {(0,): -.5, (1,): 1.6}}

        MN = dnx.markov_network(potentials)

        bqm = dnx.markov_network_bqm(MN)

        samples = dnx.sample_markov_network(MN, dimod.ExactSolver(),
                                            fixed_variables={'c': 0})

        for sample in samples:
            self.assertEqual(sample['c'], 0)

            en = 0
            for interaction, potential in potentials.items():
                config = tuple(sample[v] for v in interaction)
                en += potential[config]

            self.assertAlmostEqual(en, bqm.energy(sample))
