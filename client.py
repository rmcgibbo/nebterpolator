##############################################################################
# Imports
##############################################################################

from nebterpolator.io import XYZFile
from nebterpolator.path_operations import smooth_path
from nebterpolator.alignment import progressive_align

try:
    from mpi4py import MPI
    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

##############################################################################
# Globals
##############################################################################

in_fn = 'reaction_015.xyz'
out_fn = 'reaction_015.out.xyz'
nm_in_angstrom = 0.1
dihedral_fraction = None
smoothing_width = 20
angle_smoothing_width = 20
dihedral_smoothing_width = 20

if HAVE_MPI:
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    HAVE_MPI = (SIZE > 1)

##############################################################################
# Functions
##############################################################################

def main():
    
    if (HAVE_MPI and RANK == 0) or not HAVE_MPI:
        try:
            f = XYZFile(in_fn)
            xyzlist, atom_names = f.read_trajectory()
        finally:
            f.close()
    
        # angstroms to nm
        xyzlist *= nm_in_angstrom
    else:
        xyzlist, atom_names = None, None
    
    s_xyzlist, errors = smooth_path(xyzlist, atom_names, width=smoothing_width,
                            dihedral_fraction=dihedral_fraction,
                            angle_width=angle_smoothing_width,
                            dihedral_width=dihedral_smoothing_width)
    
    if (HAVE_MPI and RANK == 0) or not HAVE_MPI:
        try:
            f = XYZFile(out_fn, 'w')
            f.write_trajectory(s_xyzlist / nm_in_angstrom, atom_names)
        finally:
            f.close()
    
        import matplotlib.pyplot as pp
        pp.plot(errors)
        pp.plot(xyzlist[:, 10, 2])
        pp.show()


if __name__ == '__main__':
    main()
