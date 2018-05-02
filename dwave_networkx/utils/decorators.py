"""
Decorators allow for input checking and default parameter setting for
algorithms.
"""

from decorator import decorator

import dwave_networkx as dnx

__all__ = ['binary_quadratic_model_sampler']


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
    >>> import dwave_networkx as dnx
    >>> from  dwave_networkx.utils import decorators
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
    /usr/local/lib/python2.7/dist-packages/dwave_networkx/utils/decorators.pyc in _binary_quadratic_model_sampler(f, *args, **kw)
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
                default_sampler = dnx.get_default_sampler()
                if default_sampler is None:
                    raise dnx.DWaveNetworkXMissingSampler('no default sampler set')
                new_args[idx] = default_sampler
                continue

            if not hasattr(sampler, "sample_qubo") or not callable(sampler.sample_qubo):
                raise TypeError("expected sampler to have a 'sample_qubo' method")
            if not hasattr(sampler, "sample_ising") or not callable(sampler.sample_ising):
                raise TypeError("expected sampler to have a 'sample_ising' method")

        # now run the function and return the results
        return f(*new_args, **kw)
    return _binary_quadratic_model_sampler
