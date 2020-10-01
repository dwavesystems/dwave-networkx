# Copyright 2020 D-Wave Systems Inc.
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
#
# ================================================================================================
import unittest

import dwave_networkx as dnx

from dimod import ExactSolver

class TestExactCover(unittest.TestCase):
    def test_is_exact_cover(self):
        problem_set = {1, 2, 3, 4, 5}
        subsets = [{1, 2}, {3, 4, 5}]
        self.assertTrue(dnx.is_exact_cover(problem_set, subsets))

        subsets = [{1, 2, 3}, {3, 4, 5}]
        self.assertFalse(dnx.is_exact_cover(problem_set, subsets))

        problem_set = [1, 1, 2]
        subsets = [{1}, {2}]
        with self.assertRaises(ValueError):
            cover = dnx.is_exact_cover(problem_set, subsets)

    def test_basic(self):
        problem_set = {1, 2, 3, 4, 5}

        A = {1, 2}
        B = {1, 0}
        C = {3, 4, 5}
        subsets = [A,B,C]

        cover = dnx.exact_cover(problem_set, subsets, ExactSolver())
        self.assertEqual(cover, [A,C])

    def test_empty(self):
        with self.assertRaises(ValueError):
            dnx.exact_cover({}, [], ExactSolver())

        problem_set = {1,2}
        cover = dnx.exact_cover(problem_set, [], ExactSolver())
        self.assertEqual(cover, None)

    def test_no_exact_cover(self):
        A = {1, 2}
        B = {1, 0}
        C = {3, 4, 5}
        subsets = [A,B,C]

        problem_set = {1, 2, 3, 4}
        cover = dnx.exact_cover(problem_set, subsets, ExactSolver())
        self.assertEqual(cover, None)

        problem_set = {0, 1, 2}
        cover = dnx.exact_cover(problem_set, subsets, ExactSolver())
        self.assertEqual(cover, None)

    def test_invalid_problem(self):
        problem_set = [1, 1, 2, 3]
        subsets = [{1},{2},{3}]
 
        with self.assertRaises(ValueError):
            cover = dnx.exact_cover(problem_set, subsets, ExactSolver())

    def test_default_sampler(self):
        dnx.set_default_sampler(ExactSolver())
        self.assertIsNot(dnx.get_default_sampler(), None)
        
        problem_set = [1, 2, 3]
        subsets = [{1},{2},{3}]
        cover = dnx.exact_cover(problem_set, subsets)
        
        dnx.unset_default_sampler()
        with self.assertRaises(dnx.exceptions.DWaveNetworkXMissingSampler):
            cover = dnx.exact_cover(problem_set, subsets)
