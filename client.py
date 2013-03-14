##############################################################################
# Imports
##############################################################################

from nebterpolator.io import XYZFile
from nebterpolator.path_operations import smooth_path
from nebterpolator.alignment import progressive_align

##############################################################################
# Globals
##############################################################################

in_fn = 'reaction_015.xyz'
out_fn = 'reaction_015.out.xyz'
nm_in_angstrom = 0.1

##############################################################################
# Functions
##############################################################################

def main():
    try:
        f = XYZFile(in_fn)
        xyzlist, atom_names = f.read_trajectory()
    finally:
        f.close()
    
    # angstroms to nm
    xyzlist *= nm_in_angstrom
    
    s_xyzlist = smooth_path(xyzlist, atom_names, width=75,
                            dihedral_width=2.0)
    
    try:
        f = XYZFile(out_fn, 'w')
        f.write_trajectory(s_xyzlist / nm_in_angstrom, atom_names)
    finally:
        f.close()


if __name__ == '__main__':
    main()
