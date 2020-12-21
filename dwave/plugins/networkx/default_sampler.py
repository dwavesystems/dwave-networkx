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
from decorator import decorator

from dwave.plugins.networkx.exceptions import DWaveNetworkXMissingSampler

__all__ = ['set_default_sampler',
           'get_default_sampler',
           'unset_default_sampler',
           ]


_SAMPLER = None


def binary_quadratic_model_sampler(which_args):
    """Decorator to validate sampler arguments.

    Parameters
    ----------
    which_args : int or sequence of ints
        Location of the sampler arguments of the input function in the form
        `function_name(args, *kw)`. If more than one
        sampler is allowed, can be a list of locations.

    Returns
    -------
    _binary_quadratic_model_sampler : function
        Caller function that validates the sampler format. A sampler
        is expected to have `sample_qubo` and `sample_ising` methods.
        Alternatively, if no sampler is provided (or sampler is None),
        the sampler set by the `set_default_sampler` function is provided to
        the function.

    Examples
    --------
    Decorate functions like this::

        @binary_quadratic_model_sampler(1)
        def maximal_matching(G, sampler, **sampler_args):
            pass

    This example validates two placeholder samplers, which return a correct
    response only in the case of finding an independent set on a complete graph
    (where one node is always an independent set), the first valid, the second
    missing a method.

    >>> import networkx as nx
    >>> import dwave.plugins.networkx as dnx
    >>> from  dwave.plugins.networkx.utils import decorators
    >>> # Create two placeholder samplers
    >>> class WellDefinedSampler:
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
    >>> class IllDefinedSampler:
    ...     # an example sampler missing a `sample_qubo` method
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def sample_ising(self, h, J):
    ...         sample = {v: -1 for v in h}
    ...         sample[0] = 1  # set one node to true
    ...         return [sample]
    ...     def __str__(self):
    ...         return self.name
    ...
    >>> sampler1 = WellDefinedSampler('sampler1')
    >>> sampler2 = IllDefinedSampler('sampler2')
    >>> # Define a placeholder independent-set function with the decorator
    >>> @dnx.utils.binary_quadratic_model_sampler(1)
    ... def independent_set(G, sampler, **sampler_args):
    ...     Q = {(node, node): -1 for node in G}
    ...     Q.update({edge: 2 for edge in G.edges})
    ...     response = sampler.sample_qubo(Q, **sampler_args)
    ...     sample = next(iter(response))
    ...     return [node for node in sample if sample[node] > 0]
    ...
    >>> # Validate the samplers
    >>> G = nx.complete_graph(5)
    >>> independent_set(G, sampler1)
    [0]
    >>> independent_set(G, sampler2)  # doctest: +SKIP
    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    <ipython-input-35-670b71b268c7> in <module>()
    ----> 1 independent_set(G, IllDefinedSampler)
    <decorator-gen-628> in independent_set(G, sampler, **sampler_args)
    /usr/local/lib/python2.7/dist-packages/dwave.plugins.networkx/utils/decorators.pyc in _binary_quadratic_model_sampler(f, *args, **kw)
         61
         62             if not hasattr(sampler, "sample_qubo") or not callable(sampler.sample_qubo):
    ---> 63                 raise TypeError("expected sampler to have a 'sample_qubo' method")
         64             if not hasattr(sampler, "sample_ising") or not callable(sampler.sample_ising):
         65                 raise TypeError("expected sampler to have a 'sample_ising' method")
    TypeError: expected sampler to have a 'sample_qubo' method

    """
    @decorator
    def _binary_quadratic_model_sampler(f, *args, **kw):
        # convert into a sequence if necessary
        if isinstance(which_args, int):
            iter_args = (which_args,)
        else:
            iter_args = iter(which_args)

        # check each sampler for the correct methods
        new_args = [arg for arg in args]
        for idx in iter_args:
            sampler = args[idx]

            # if no sampler is provided, get the default sampler if it has
            # been set
            if sampler is None:
                # this sampler has already been vetted
                default_sampler = get_default_sampler()
                if default_sampler is None:
                    raise DWaveNetworkXMissingSampler('no default sampler set')
                new_args[idx] = default_sampler
                continue

            if not hasattr(sampler, "sample_qubo") or not callable(sampler.sample_qubo):
                raise TypeError("expected sampler to have a 'sample_qubo' method")
            if not hasattr(sampler, "sample_ising") or not callable(sampler.sample_ising):
                raise TypeError("expected sampler to have a 'sample_ising' method")

        # now run the function and return the results
        return f(*new_args, **kw)
    return _binary_quadratic_model_sampler


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
