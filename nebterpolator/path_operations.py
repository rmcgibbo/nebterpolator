"""Higher-level methods for manipulating trajectories
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np
import itertools
from mpi4py import MPI
from scipy.interpolate import UnivariateSpline

# local imports
import core
from mpiutils import mpi_root, group, interweave
from alignment import align_trajectory
from smoothing import buttersworth_smooth, angular_smooth, window_smooth
from inversion import least_squares_cartesian

##############################################################################
# Globals
##############################################################################

__all__ = ['smooth_internal', 'smooth_cartesian', 'union_connectivity']
DEBUG = True

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


##############################################################################
# Functions
##############################################################################


def smooth_internal(xyzlist, atom_names, width, **kwargs):
    """Smooth a trajectory by transforming to redundant, internal coordinates,
    running a 1d timeseries smoothing algorithm on each DOF, and then
    reconstructing a set of consistent cartesian coordinates.

    TODO: write this function as a iterator that yields s_xyz, so that
    they can be saved to disk (async) immediately when they're produced.

    Parameters
    ----------
    xyzlist : np.ndarray
        Cartesian coordinates
    atom_names : array_like of strings
        The names of the atoms. Required for determing connectivity.
    width : float
        Width for the smoothing kernels

    Other Parameters
    ----------------
    bond_width : float
        Override width just for the bond terms
    angle_width : float
        Override width just for the angle terms
    dihedral_width : float
        Override width just for the dihedral terms
    xyzlist_guess : np.ndarray
        Cartesian coordinates to use as a guess during the
        reconstruction from internal


    Returns
    -------
    smoothed_xyzlist : np.ndarray
    """
    bond_width = kwargs.pop('bond_width', width)
    angle_width = kwargs.pop('angle_width', width)
    dihedral_width = kwargs.pop('dihedral_width', width)
    xyzlist_guess = kwargs.pop('xyzlist_guess', xyzlist)
    for key in kwargs.keys():
        raise KeyError('Unrecognized key, %s' % key)

    ibonds, iangles, idihedrals = None, None, None
    s_bonds, s_angles, s_dihedrals = None, None, None
    with mpi_root():
        ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names)
        xyzlist *= 18.903
        # get the internal coordinates in each frame
        bonds = core.bonds(xyzlist, ibonds)
        angles = core.angles(xyzlist, iangles)
        dihedrals = core.dihedrals(xyzlist, idihedrals)

        # run the smoothing
        s_bonds = np.zeros_like(bonds)
        s_angles = np.zeros_like(angles)
        s_dihedrals = np.zeros_like(dihedrals)
        for i in xrange(bonds.shape[1]):
            #s_bonds[:, i] = buttersworth_smooth(bonds[:, i], width=bond_width)
            s_bonds[:, i] = window_smooth(bonds[:, i], window_len=bond_width)
        for i in xrange(angles.shape[1]):
            #s_angles[:, i] = buttersworth_smooth(angles[:, i], width=angle_width)
            s_angles[:, i] = window_smooth(angles[:, i], window_len=angle_width)
        # filter the dihedrals with the angular smoother, that filters
        # the sin and cos components separately
        for i in xrange(dihedrals.shape[1]):
            #s_dihedrals[:, i] = angular_smooth(dihedrals[:, i],
            #    smoothing_func=buttersworth_smooth, width=dihedral_width)
            s_dihedrals[:, i] = angular_smooth(dihedrals[:, i],
                                               smoothing_func=window_smooth, 
                                               window_len=dihedral_width)

        # group these into SIZE components, to be scattered
        xyzlist_guess = group(xyzlist_guess, SIZE)
        s_bonds = group(s_bonds, SIZE)
        s_angles = group(s_angles, SIZE)
        s_dihedrals = group(s_dihedrals, SIZE)

    if RANK != 0:
        xyzlist_guess = None

    # scatter these
    xyzlist_guess = COMM.scatter(xyzlist_guess, root=0)
    s_bonds = COMM.scatter(s_bonds, root=0)
    s_angles = COMM.scatter(s_angles, root=0)
    s_dihedrals = COMM.scatter(s_dihedrals, root=0)

    # broadcast the indices to every node
    ibonds = COMM.bcast(ibonds, root=0)
    iangles = COMM.bcast(iangles, root=0)
    idihedrals = COMM.bcast(idihedrals, root=0)

    # compute the inversion for each frame
    s_xyzlist = np.zeros_like(xyzlist_guess)
    errors = np.zeros(len(xyzlist_guess))
    for i, xyz_guess in enumerate(xyzlist_guess):
        # if i > 0:
        #     xyz_guess = s_xyzlist[i-1]
        r = least_squares_cartesian(s_bonds[i], ibonds, s_angles[i], iangles,
                                    s_dihedrals[i], idihedrals, xyz_guess)
        s_xyzlist[i], errors[i] = r
        s_xyzlist[i] /= 18.903
        print 'Rank %2d: (%3d)->xyz: error %f' % (RANK,
                    RANK + i*SIZE, errors[i])

    # gather the results back on root
    s_xyzlist = COMM.gather(s_xyzlist, root=0)
    errors = COMM.gather(errors, root=0)

    return_value = (None, None)
    with mpi_root():
        # interleave the results back together
        return_value = (interweave(s_xyzlist), interweave(errors))

    return return_value


def smooth_cartesian(xyzlist, strength=None, weights=None):
    """Smooth cartesian coordinates with spline on each coordinate

    Parameters
    ----------
    xyzlist : np.ndarray, ndim=3, shape=[n_frames, n_atoms, n_dims]
        The cartesian coordinates
    strength : float, optional
        Positive smoothing factor used to choose the number of knots.
    errors : np.ndarray, ndim=1, shape=[n_frames]
        The weight to assign to each point
    Notes
    -----
    The underlying spline engine is in scipy. It's documentation is here at
    the url below.
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    """

    if not xyzlist.ndim == 3:
        raise TypeError('xyzlist must be 3d')
    n_frames, n_atoms, n_dims = xyzlist.shape
    if weights is None:
        weights = np.ones(n_frames, dtype=np.float)
    if not len(weights) == n_frames:
        raise ValueError('errors must be of length n_frames')
    weights /= sum(weights)

    aligned = align_trajectory(xyzlist, 'progressive')
    smoothed = np.empty_like(xyzlist)

    t = np.arange(n_frames)

    for i in range(n_atoms):
        for j in range(n_dims):
            y = aligned[:, i, j]
            smoothed[:, i, j] = UnivariateSpline(x=t, y=y,
                                    s=strength, w=weights)(t)

    return align_trajectory(smoothed, which='progressive')


def union_connectivity(xyzlist, atom_names):
    """Get the union of all possible proper bonds/angles/dihedrals
    that appear in any frame of a trajectory

    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The cartesian coordinates of each frame in a trajectory. The units
        should be in nanometers.
    atom_names : array_like of strings, length=n_atoms
        A list of the names of each of the atoms, which will be used for
        grabbing the covalent radii.
    """
    if not xyzlist.ndim == 3:
        raise ValueError('I need an xyzlist that is 3d')

    # collect a set of all of the bonds/angles/dihedrals that appear in any
    # frame. These will be sets of tuples -- 2-tuples for bonds, 3-tuples for
    # angles, and 4-tuples for dihedrals. We need to use tuples instead of
    # numpy arrays so that they're immutable, which lets the set data
    # structure make sure they're unique
    set_bonds = set()
    set_angles = set()
    set_dihedrals = set()

    for xyz in xyzlist:
        bonds = core.bond_connectivity(xyz, atom_names)
        angles = core.angle_connectivity(bonds)
        dihedrals = core.dihedral_connectivity(bonds)

        set_bonds.update(set([tuple(e) for e in bonds]))
        set_angles.update(set([tuple(e) for e in angles]))
        set_dihedrals.update(set([tuple(e) for e in dihedrals]))

    # sort the sets and convert them to the numpy arrays that we
    # prefer. The sorting is not strictly necessary, but it seems good
    # form.

    ibonds = np.array(sorted(set_bonds, key=lambda e: sum(e)))
    iangles = np.array(sorted(set_angles, key=lambda e: sum(e)))
    idihedrals = np.array(sorted(set_dihedrals, key=lambda e: sum(e)))

    # get ALL of the possible bonds
    # ibonds = np.array(list(itertools.combinations(range(xyzlist.shape[1]), 2)))

    return ibonds, iangles, idihedrals
