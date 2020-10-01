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
from dwave_networkx.utils import binary_quadratic_model_sampler
from dimod import BinaryQuadraticModel

__all__ = ['is_exact_cover', 'exact_cover']

def is_exact_cover(problem_set, subsets):
    """Determines whether the given list of subsets is an exact cover of
    `problem_set`.

    An exact cover is a collection of subsets of `problem_set` that contain every 
    element in `problem_set` exactly once. More formally, all elements of the 
    exact cover are disjoint and the union of elements is the original problem_set. 

    Parameters
    ----------
    problem_set : iterable
        An iterable of unique numbers

    subsets : list(iterable(numeric))
        A list of subsets of `problem_set`  

    Returns
    -------
    is_cover : bool
        True if the given list of subsets forms an exact cover.

    Examples
    --------
    This example checks two potential covers for a problem set of 
    {1, 2, 3, 4, 5}.

    >>> import dwave_networkx as dnx
    >>> problem_set = {1, 2, 3, 4, 5}
    >>> subsets = [{1, 2}, {3, 4, 5}]
    >>> dnx.is_exact_cover(problem_set, subsets)
    True
    >>> subsets = [{1, 2, 3}, {3, 4, 5}]
    >>> dnx.is_exact_cover(problem_set, subsets)
    False

    """
    # Verify that problem_set is valid
    orig_len = len(problem_set)
    problem_set = set(problem_set)
    if len(problem_set) != orig_len:
        raise ValueError("problem_set should not contain any duplicates")
    
    # Verify that every element in problem_set is contained exactly once in subsets
    cover = [x for subset in subsets for x in subset]

    if len(cover) != orig_len:
        return False

    return all(x in cover for x in problem_set)

@binary_quadratic_model_sampler(2)
def exact_cover(problem_set, subsets, sampler=None, **sampler_args):  
    """Returns an exact cover.
 
    An exact cover is a collection of subsets of `problem_set` that contain every 
    element in `problem_set` exactly once. More formally, all elements of the 
    exact cover are disjoint and the union of elements is the original problem_set. 

    This function defines a QUBO with ground states corresponding to an exact
    cover of the problem set, if it exists, and uses the sampler to sample from
    it.

    Parameters
    ----------
    problem_set : iterable
        An iterable of unique numbers

    subsets : list(iterable(numeric))
        A list of subsets of `problem_set` used to find an exact cover.

    sampler
        A binary quadratic model sampler. A sampler is a process that samples
        from low energy states in models defined by an Ising equation or a
        Quadratic Unconstrained Binary Optimization Problem (QUBO). A sampler is
        expected to have a 'sample_qubo' and 'sample_ising' method. A sampler is
        expected to return an iterable of samples, in order of increasing
        energy. If no sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    exact_cover : list(iterable(numeric))/None
       List of subsets that form an exact cover, or None if an exact cover does
       not exist.

    Examples
    --------
    This example uses a sampler from `dimod
    <https://github.com/dwavesystems/dimod>`_ to find an exact cover of the set
    {1, 2, 3, 4, 5, 6}, given a list of some of its subsets.

    >>> import dwave_networkx as dnx
    >>> import dimod
    >>> sampler = dimod.ExactSolver()  # small testing sampler
    >>> problem_set = {1, 2, 3, 4, 5, 6}
    >>> subsets = [{1, 2, 3}, {1, 4, 5}, {5}, {4, 6}, {6}]
    >>> dnx.exact_cover(problem_set, subsets, sampler)
    [{1, 2, 3}, {5}, {4, 6}]

    References
    ----------
    Based on the formulation presented in [AL]_

    .. [AL] Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, Volume 2, Article 5.

    """
    if not problem_set:
        raise ValueError("problem_set should not be empty")

    if not subsets:
        return None

    # Check that problem set is valid
    orig_len = len(problem_set)
    problem_set = set(problem_set)
    if len(problem_set) != orig_len:
        raise ValueError("problem_set should not contain any duplicates")

    # Create BQM
    bqm = BinaryQuadraticModel({}, {}, 0, 'BINARY')

    offset = 0
    for element in problem_set:
        offset += 1

        for i in range(len(subsets)):      
            if element in subsets[i]:
                bqm.add_variable(i, -1)

                for j in range(i-1, -1, -1):
                    if element in subsets[j]:
                        bqm.add_interaction(i, j, 2)

    bqm.offset = offset
    
    response = sampler.sample(bqm)

    # Get lowest energy sample
    sample = next(iter(response))

    result = []
    for subset_index in sample:
        if sample[subset_index] > 0:
            result.append(subsets[subset_index])

    # Validate that result is actually an exact cover
    if is_exact_cover(problem_set, result):
        return result
    else:
        return None
