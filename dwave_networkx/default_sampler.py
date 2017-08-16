"""Allows the user to specify a binary quadratic model sampler in a single place.

API Definition
--------------

A sampler is a process that samples
from low energy states in models defined by an Ising equation
or a Quadratic Unconstrained Binary Optimization Problem
(QUBO). A sampler is expected to have a 'sample_qubo' and
'sample_ising' method. A sampler is expected to return an
iterable of samples, in order of increasing energy.

Example sampler
---------------
First we can create a stand in for a binary quadratic model sampler that can be
used in the following examples.

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

No default sampler set
----------------------
If the user wishes to specify which sampler each binary quadratic model sampler
algorithm uses, the user can provide the sampler directly to
the functions.

>>> sampler = ExampleSampler('sampler')
>>> G = nx.complete_graph(5)
>>> dnx.maximum_independent_set(G, sampler)
[0]

Setting a default sampler
-------------------------
Alternatively, the user can specify a sampler that will be used
by default.

>>> sampler0 = ExampleSampler('sampler0')
>>> dnx.set_default_sampler(sampler0)

This sampler will now be used by any function where no sampler
is specified

>>> dnx.maximum_independent_set(G)
[0]

A different sampler can still be provided, in which case the
provided sampler will be used instead of the default.

>>> sampler1 = ExampleSampler('sampler1')
>>> dnx.set_default_sampler(sampler0)
>>> dnx.maximum_independent_set(G, sampler1)
[0]

Unsetting a default sampler
---------------------------
The user can also unset the default.

>>> dnx.unset_default_sampler()
>>> print(dnx.get_default_sampler())
None

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
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy.

    Examples
    --------
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
    >>> dnx.set_default_sampler(sampler0)  # doctest: +SKIP
    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    'sampler0'
    >>> dnx.unset_default_sampler()  # doctest: +SKIP
    >>> print(dnx.get_default_sampler())
    None
    """
    global _SAMPLER
    _SAMPLER = None


def get_default_sampler():
    """Gets the current default sampler.

    Examples
    --------
    >>> print(dnx.get_default_sampler())
    None
    >>> dnx.set_default_sampler(sampler)
    >>> print(dnx.get_default_sampler())  # doctest: +SKIP
    'sampler'

    """
    return _SAMPLER
