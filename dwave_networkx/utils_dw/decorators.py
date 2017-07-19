from decorator import decorator

__all__ = ['discrete_model_sampler']


def discrete_model_sampler(which_args):
    """Decorator to check sampler arguments.

    Parameters
    ----------
    which_args : int or sequence of ints
        Location of the sampler arguments in args. If more than one
        sampler is allowed, can be a list of locations.

    Returns
    -------
    _discrete_model_sampler : function
        Function which checks the sampler for correctness.

    Examples
    --------
    Decorate functions like this::

    @discrete_model_sampler(1)
    def maximal_matching(G, sampler, **sampler_args):
        pass
    """
    @decorator
    def _discrete_model_sampler(f, *args, **kw):

        # convert into a sequence if necessary
        if isinstance(which_args, int):
            which_args = (which_args,)

        # check each sampler for the correct methods
        for idx in iter(which_args):
            sampler = args[idx]

            if not hasattr(sampler, "sample_qubo") or not callable(sampler.sample_qubo):
                raise TypeError("expected sampler to have a 'sample_qubo' method")
            if not hasattr(sampler, "sample_ising") or not callable(sampler.sample_ising):
                raise TypeError("expected sampler to have a 'sample_ising' method")

        # now run the function and return the results
        return f(*args, **kw)
    return _discrete_model_sampler
