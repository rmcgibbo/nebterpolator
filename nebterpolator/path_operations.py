"""Higher-level methods for manipulating trajectories
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np

# local imports
import core
from align import progressive_align
from smoothing import filtfit_smooth
from inversion import least_squares_cartesian

##############################################################################
# Globals
##############################################################################

__all__ = ['smooth_trajectory', 'union_connectivity']
DEBUG = True

##############################################################################
# Functions
##############################################################################


def smooth_path(xyzlist, atom_names, width, **kwargs):
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
        
    Returns
    -------
    smoothed_xyzlist : np.ndarray
    """
    
    bond_width = kwargs.pop('bond_width', width)
    angle_width = kwargs.pop('angle_width', width)
    dihedral_width = kwargs.pop('dihedral_width', width)
    for key in kwargs.keys():
        print ('WARNING: Ignoring key "%s" in kwarg. '
               'It wasn\'t recognized.' % key)
    
    ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names)

    # get the internal coordinates in each frame
    bonds = core.bonds(xyzlist, ibonds)
    angles = core.angles(xyzlist, iangles)
    dihedrals = core.dihedrals(xyzlist, idihedrals)
    
    
    # smooth the timeseries of internal coordinates
    s_bonds = np.zeros_like(bonds)
    s_angles = np.zeros_like(angles)
    s_dihedrals = np.zeros_like(dihedrals)
    for i in xrange(bonds.shape[1]):
        s_bonds[:, i] = filtfit_smooth(bonds[:, i], width=bond_width)
    for i in xrange(angles.shape[1]):
        s_angles[:, i] = filtfit_smooth(angles[:, i], width=angle_width)
    for i in xrange(dihedrals.shape[1]):
        s_dihedrals[:, i] = filtfit_smooth(dihedrals[:, i], width=dihedral_width)
    
    
    # reconstruct the cartesian coordinates
    s_xyzlist = np.zeros_like(xyzlist)
    for i, xyz in enumerate(xyzlist):
        print 'reconstructing frame %d of %d' % (i, len(xyzlist))
        s_xyzlist[i] = least_squares_cartesian(s_bonds[i], ibonds, s_angles[i],
                                iangles, s_dihedrals[i], idihedrals, xyz,
                                display=True)
    return progressive_align(s_xyzlist)

def plot_smoothing(bonds, s_bonds, angles, s_angles, dihedrals, s_dihedrals):
    """
    Debuggin method to plot a few random bond/angle/dihedral timeseries.
    This can use used to figure out what smoothing width is appropriate for
    your dataset.
    
    """
    import matplotlib.pyplot as pp
    
    pp.subplot(3,1,1)
    bond_i = np.random.randint(len(bonds[0]))
    pp.title("Bond %d over timeseries" % bond_i)
    pp.plot(s_bonds[:, bond_i], label='smoothed')
    pp.plot(bonds[:, bond_i], label='raw data')
    pp.xlabel('Time')
    pp.ylabel('Bond %d' % bond_i)
    
    pp.subplot(3,1,2)
    angle_i = np.random.randint(len(angles[0]))
    pp.title("Angle %d over timeseries" % angle_i)
    pp.plot(s_angles[:, angle_i], label='smoothed')
    pp.plot(angles[:, angle_i], label='raw data')
    pp.xlabel('Time')
    pp.ylabel('Angle %d' % bond_i)
    
    pp.subplot(3,1,3)
    dihedral_i = np.random.randint(len(dihedrals[0]))
    pp.title("Dihedral %d over timeseries" % dihedral_i)
    pp.plot(s_dihedrals[:, dihedral_i], label='smoothed')
    pp.plot(dihedrals[:, dihedral_i], label='raw data')
    pp.xlabel('Time')
    pp.ylabel('Dihedral %d' % dihedral_i)
    
    pp.subplots_adjust(hspace=0.5)
    pp.show()
        

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

    return ibonds, iangles, idihedrals
