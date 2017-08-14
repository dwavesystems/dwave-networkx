"""
Decorators allow for input checking and default parameter setting for
algorithms.
"""

from decorator import decorator

import dwave_networkx as dnx

__all__ = ['binary_quadratic_model_sampler']


def binary_quadratic_model_sampler(which_args):
    """Decorator to check sampler arguments.

    Parameters
    ----------
    which_args : int or sequence of ints
        Location of the sampler arguments in args. If more than one
        sampler is allowed, can be a list of locations.

    Returns
    -------
    _binary_quadratic_model_sampler : function
        Function which checks the sampler for correctness. A sampler
        is expected to have "sample_qubo" and "sample_ising" methods.
        Alternatively, if no sampler is provided (or sampler is None)
        the sampler as set by set_default_sampler will be provided to
        the function.

    Examples
    --------
    Decorate functions like this::

        @binary_quadratic_model_sampler(1)
        def maximal_matching(G, sampler, **sampler_args):
            pass
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
