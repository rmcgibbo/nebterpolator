"""


"""

##############################################################################
# Imports
##############################################################################

import numpy as np
from .connectivity import bond_connectivity

##############################################################################
# Functions
##############################################################################


def to_interal(xyzlist, atom_names):
    """
    
    
    
    """
    # collect a set of all of the bonds/angles/dihedrals that appear in any
    # frame. These will be sets of tuples -- 2-tuples for bonds, 3-tuples for
    # angles, and 4-tuples for dihedrals. We need to use tuples instead of
    # numpy arrays so that they're immutable, which lets the set data
    # structure make sure they're unique
    set_bonds = set()
    set_angles = set()
    set_dihedrals = set()
    
    for xyz in xyzlist:
        bonds = bond_connectivity(xyz, atom_names)
        angles = angle_connectivity(bonds)
        dihedrals = dihedral_connectivity(angles)
        
        set_bonds.update(set([tuple(e) for e in bonds]))
        set_angles.update(set([tuple(e) for e in angles]))
        set_dihedrals.update(set([tuple(e) for e in dihedrals]))
        
    
