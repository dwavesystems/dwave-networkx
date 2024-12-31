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

"""
Decorators allow for input checking and default parameter setting for
algorithms.
"""

import functools
import inspect

import dwave_networkx as dnx

__all__ = ['binary_quadratic_model_sampler', 'ImplementationHook']


def binary_quadratic_model_sampler(which_args):
    """Decorator to validate sampler arguments.

    Parameters
    ----------
    which_args : int or sequence of ints
        Location of the sampler arguments of the input function in the form
        `function_name(args, *kw)`. If more than one
        sampler is allowed, can be a list of locations.

    """
    def decorator(f):
        @functools.wraps(f)
        def func(*args, **kwargs):
            bound_arguments = inspect.signature(f).bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            args = bound_arguments.args
            kw = bound_arguments.kwargs

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
        return func
    return decorator


class ImplementationHook:
    """A decorator class to provide a temporary slot value which overwrites itself

    This is used by the ``dwave_networkx.TopologyFamily`` enum.  Each family
    object holds references to specific implementations of generic functions.
    But to avoid circular dependencies, we first install this hook when the
    family object is constructed.  Then, when each generic function is
    implemented, that implementation gets decorated with the ``implementation``
    method of the respective ``ImplementationHook`` object.  When the decoration
    occurs, the hook replaces itself with the decorated function.  The decorated
    function itself is left unchanged.

    For example, in ``dwave_networkx.drawing.chimera_layout``, we implement the
    function ``draw_chimera_embedding``.  We also want to make an alias of that
    named ``dwave_networkx.CHIMERA.draw_embedding``.  So, in the construction
    of ``dwave_networkx.CHIMERA`` we have put an ``ImplementationHook`` in place
    of ``draw_embedding``.  Then, when we implement ``draw_chimera_embedding``,
    we write

        @CHIMERA.draw_embedding.implementation
        def draw_chimera_embedding(...)
            ...

    which provides the desired alias.

    """

    def __init__(self, obj, name):
        self.obj = obj
        self.name = name

    def implementation(self, f):
        # monkeypatch the object with this implementation
        setattr(self.obj, self.name, f)

        # don't forget to return f, or the decorated function will be None
        return f

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"the {self.name} method of {self.obj:r} has not been attached"
        )
