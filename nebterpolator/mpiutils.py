"""Utilities for working with MPI
"""

##############################################################################
# Imports
##############################################################################

import sys
import inspect

__all__ = ['mpi_root', 'mpi_rank', 'SelectiveExecution']

##############################################################################
# Code
##############################################################################


def mpi_rank(comm=None):
    """Get the rank of the curent MPI node

    Parameters
    ----------
    comm : mpi communicator, optional
        The MPI communicator. By default, we use the COMM_WORLD

    Returns
    -------
    rank : int
        The rank of the current mpi process. The return value is 0
        (the root node) if MPI is not running
    """
    if comm is None:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
        except ImportError:
            rank = 0
    else:
        rank = comm.Get_rank()

    return rank


class _ExitContext(Exception):
    """Special exception used to skip execution of a with-statement block."""
    pass


class SelectiveExecution(object):
    """Contect manager that executes body only under a certain
    condition.

    http://stackoverflow.com/questions/12594148
    """

    def __init__(self, skip=False):
        """Create the context manager

        Parameters
        ----------
        skip : bool
            If true, the body will be skipped
        """
        self.skip = skip

    def __enter__(self):
        if self.skip:
            # Do some magic
            sys.settrace(lambda *args, **keys: None)
            frame = inspect.currentframe(1)
            frame.f_trace = self.trace

    def trace(self, frame, event, arg):
        raise _ExitContext()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is _ExitContext:
            return True
        else:
            return False


class mpi_root(SelectiveExecution):
    """Context manager that selectively executes its body on the root node"""
    def __init__(self, comm=None):
        """Open a context managert that only executes its body on the root
        node.

        Paramters
        ---------
        comm : mpi communicator, optional
            The MPI communicator. By default, we use the COMM_WORLD
        """
        skip_if = (mpi_rank(comm) != 0)
        super(mpi_root, self).__init__(skip_if)


##############################################################################
# Test code
##############################################################################


def main():
    "Example code"
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    a = None
    with mpi_root():
        print 'Initializing a only on rank=%s' % rank
        a = [1, 2, 3, 4]
    a = comm.bcast(a)

    print 'RANK %s, a=%s' % (rank, a)


if __name__ == '__main__':
    main()
