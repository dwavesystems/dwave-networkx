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

"""Sets a binary quadratic model sampler used by default
for functions that require a sample when none is specified.

A sampler is a process that samples
from low-energy states in models defined by an Ising equation
or a Quadratic Unconstrained Binary Optimization Problem
(QUBO).

Sampler API
-----------
* Required Methods: 'sample_qubo' and 'sample_ising'
* Return value: iterable of samples, in order of increasing energy

See  `dimod <https://github.com/dwavesystems/dimod>`_ for details.

Example
-------
This example creates and uses a placeholder for binary quadratic model
samplers that returns a correct response only in the case of finding an
independent set on a complete graph (where one node is always an
independent set). The placeholder sampler can be used to test the simple
examples of the functions for configuring a default sampler.

>>> # Create a placeholder sampler
>>> class ExampleSampler:
...     # an example sampler, only works for independent set on complete
...     # graphs
...     def __init__(self, name):
...         self.name = name
...     def sample_ising(self, h, J):
...         sample = {v: -1 for v in h}
...         sample[0] = 1  # set one node to true
...         return [sample]
...     def sample_qubo(self, Q):
...         sample = {v: 0 for v in set().union(*Q)}
...         sample[0] = 1  # set one node to true
...         return [sample]
...     def __str__(self):
...         return self.name
...
>>> # Identify the new sampler as the default sampler
>>> sampler0 = ExampleSampler('sampler0')
>>> dnx.set_default_sampler(sampler0)
>>> # Find an independent set using the default sampler
>>> G = nx.complete_graph(5)
>>> dnx.maximum_independent_set(G)
[0]

"""

from dwave_networkx.utils.decorators import binary_quadratic_model_sampler

__all__ = ['set_default_sampler', 'get_default_sampler', 'unset_default_sampler']


_SAMPLER = None


@binary_quadratic_model_sampler(0)
def set_default_sampler(sampler):
    """Sets a default binary quadratic model sampler.

    Parameters
    ----------
    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low-energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy.

    Examples
    --------
    This example sets sampler0 as the default sampler and finds an independent
    set for graph G, first using the default sampler and then overriding it by
    specifying a different sampler.

    >>> dnx.set_default_sampler(sampler0)  # doctest: +SKIP
    >>> indep_set = dnx.maximum_independent_set_dm(G)  # doctest: +SKIP
    >>> indep_set = dnx.maximum_independent_set_dm(G, sampler1)  # doctest: +SKIP

    """
    global _SAMPLER
    _SAMPLER = sampler


def unset_default_sampler():
    """Resets the default sampler back to None.

    Examples
    --------
    This example sets sampler0 as the default sampler, verifies the setting,
    then resets the default, and verifies the resetting.

    >>> dnx.set_default_sampler(sampler0)  # doctest: +SKIP
    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    'sampler0'
    >>> dnx.unset_default_sampler()  # doctest: +SKIP
    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    None
    """
    global _SAMPLER
    _SAMPLER = None


def get_default_sampler():
    """Queries the current default sampler.

    Examples
    --------
    This example queries the default sampler before and after specifying
    a default sampler.

    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    None
    >>> dnx.set_default_sampler(sampler)  # doctest: +SKIP
    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    'sampler'

    """
    return _SAMPLER
