#!/usr/bin/env python

##############################################################################
# Imports
##############################################################################

from nebterpolator.io import XYZFile
from nebterpolator.path_operations import smooth_internal, smooth_cartesian
import os, sys, argparse

#from nebterpolator.alignment import align_trajectory
#import matplotlib.pyplot as pp

##############################################################################
# Globals
##############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--morse', type=float, default=0.0, help='Weight of Morse potential')
parser.add_argument('--width', type=int, default=-1, help='Smoothing width, default is to use the trajectory length')
args, sys.argv= parser.parse_known_args(sys.argv)

input_filename = sys.argv[1]
if len(sys.argv) > 2:
    output_filename = sys.argv[2]
else:
    output_filename = os.path.splitext(sys.argv[1])[0]+"_out.xyz"
    print "Supplying default output filename", output_filename
nm_in_angstrom = 0.1

# these two parameters are adjustable, and depend on the length of the traj

# cutoff period for the internal coordinate smoother. motions with a shorter
# period than this (higher frequency) will get filtered out
smoothing_width = args.width

# the spline smoothing factor used for the cartesian smoothing step, that
# runs after the internal coordinates smoother. The point of this is ONLY
# to correct for "jitters" in the xyz coordinates that are introduced by
# imperfections in the redundant internal coordinate -> xyz coordinate
# step, which runs after smoothing in internal coordinates
xyz_smoothing_strength = 2.0
FinalSmooth = True
if not FinalSmooth:
    xyz_smoothing_strength = 0.0

##############################################################################
# Script
##############################################################################

xyzlist, atom_names = None, None
with XYZFile(input_filename) as f:
    xyzlist, atom_names = f.read_trajectory()
    # angstroms to nm
    xyzlist *= nm_in_angstrom
if xyzlist.shape[1] < 4:
    print "Interpolator cannot handle less than four atoms."
    sys.exit()

# If smoothing width not provided, default to using the trajectory length
if smoothing_width == -1:
    smoothing_width = len(xyzlist)
    smoothing_width += smoothing_width%2
    smoothing_width -= 1
if smoothing_width%2 != 1:
    raise RuntimeError("Smoothing width must be an odd number")

# transform into redundant internal coordinates, apply a fourier based
# smoothing, and then transform back to cartesian.
# the internal -> cartesian bit is the hard step, since there's no
# guarentee that a set of cartesian coordinates even exist that satisfy
# the redundant internal coordinates, after smoothing.

# we're using a levenberg-marquardt optimizer to find the "most consistent"
# cartesian coordinates

# currently, the choice of what internal coordinates to use is buried
# a little in the code, in the function path_operations.union_connectivity
# basically, we're using ALL pairwise distances, all of the angles between
# sets of three atoms, a-b-c, that actually get "bonded" during the
# trajectory, and all of the dihedral angles between sets of 4 atoms,
# a-b-c-d, that actually get "bonded" during the trajectory.
smoothed, errors = smooth_internal(xyzlist, atom_names, width=smoothing_width, bond_width=smoothing_width, angle_width = smoothing_width, dihedral_width = smoothing_width, w_morse = args.morse)


print 'Saving output to', output_filename
# apply a bit of spline smoothing in cartesian coordinates to
# correct for jitters
jitter_free = smooth_cartesian(smoothed,
                               strength=xyz_smoothing_strength,
                               weights=1.0/errors)
with XYZFile(output_filename, 'w') as f:
    f.write_trajectory(jitter_free / nm_in_angstrom, atom_names)
# else:
#     with XYZFile(output_filename, 'w') as f:
#         f.write_trajectory(smoothed / nm_in_angstrom, atom_names)
        
