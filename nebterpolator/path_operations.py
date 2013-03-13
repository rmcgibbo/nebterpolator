"""Higher-level methods for manipulating trajectories
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np

# local imports
from core import connectivity

##############################################################################
# Functions
##############################################################################

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
        bonds = connectivity.bond_connectivity(xyz, atom_names)
        angles = connectivity.angle_connectivity(bonds)
        dihedrals = connectivity.dihedral_connectivity(bonds)

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