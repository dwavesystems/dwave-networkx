from decorator import decorator

__all__ = ['quantum_annealer_solver']


def quantum_annealer_solver(which_arg):
    """Decorator to check that the provided quantum annealer has the
    expected form.

    Parameters
    ----------
    which_arg : int
        Location of the quantum_annealer object in args.

    Returns
    -------
    _quantum_annealer_solver : function
        Function which checks the annealer

    """
    @decorator
    def _quantum_annealer_solver(f, *args, **kw):

        solver = args[which_arg]

        if not hasattr(solver, "sample_qubo") \
                or not callable(solver.sample_qubo):
            raise TypeError("expected solver to have a 'sample_qubo' method")

        return f(*args, **kw)
    return _quantum_annealer_solver
