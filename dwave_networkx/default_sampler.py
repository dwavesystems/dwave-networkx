from dwave_networkx.utils_dw.decorators import discrete_model_sampler

__all__ = ['set_default_sampler', 'get_default_sampler', 'unset_default_sampler']


_SAMPLER = None


@discrete_model_sampler(0)
def set_default_sampler(sampler):
    """Sets a default discrete model sampler.

    Parameters
    ----------
    sampler
        A discrete model sampler. A sampler is a process that samples
        from low energy states in models defined by an Ising equation
        or a Quadratic Unconstrainted Binary Optimization Problem
        (QUBO). A sampler is expected to have a 'sample_qubo' and
        'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy.

    Examples
    --------
    >>> dnx.set_default_sampler(sampler0)
    >>> indep_set = dnx.maximum_independent_set_dm(G)  # uses sampler0
    >>> indep_set = dnx.maximum_independent_set_dm(G, sampler1)

    """
    global _SAMPLER
    _SAMPLER = sampler


def unset_default_sampler():
    """Resets the default sampler back to None.

    Examples
    --------
    >>> dnx.set_default_sampler(sampler0)
    >>> print(dnx.get_default_sampler())
    'sampler0'
    >>> dnx.unset_default_sampler()
    >>> print(dnx.get_default_sampler())
    'None'
    """
    global _SAMPLER
    _SAMPLER = None


def get_default_sampler():
    """Gets the current default sampler.

    Examples
    --------
    >>> print(dnx.get_default_sampler())
    'None'
    >>> dnx.set_default_sampler(sampler0)
    >>> print(dnx.get_default_sampler())
    'sampler0'

    """
    return _SAMPLER
