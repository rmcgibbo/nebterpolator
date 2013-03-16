nebterpolator
=============

A library for generating smooth paths from ab-inito MD trajectories. These
paths might serve as input for an NEB calculation. The key component of this
package is the ability to translate from cartesian coordinates to redundant
internal coordinates (bond lengths, angles, dihedrals) and to go the other
way, from redundant internal coordinates to an "optimal" set of cartesian
coordinates.

Requirements
------------
- python 2.6+  (tested with 2.7.3)
- numpy, 1.6+  (tested with 1.6.2)
- scipy, 0.10+ (tested with 0.10.1)
- mpi4py 1.3+  (tested with 1.3)

Getting Started
---------------

Open up the file `client.py` and take a look. Try running it!

`python client.py`

or

`mpirun -np 2 client.py`


Algorithm
---------

- Select a set of redundant internal coordinates
  - currently, we're using ALL pairwise distances, all of the angles between
    sets of three atoms, a-b-c, that actually get "bonded" during the
    trajectory, and all of the dihedral angles between sets of 4 atoms,
    a-b-c-d, that actually get "bonded" during the trajectory. I observed
    empirically that using ALL pairwise distances seems to work better than
    only using pairs corresponding to actually bonded atoms.
- Transform into the internal coordinate space
- Apply smoothing filter to the timeseries from each internal coordinate
  degree of freedom independently
   - The cutoff frequency is _the_ key variable
- For each frame, find a set of xyz coordinates that are optimally consistent
  with the smoothed internal coordinates
   - This is a computationally intensive optimization problem, and it's
     parallelized over MPI.
- Apply a very slight smoothing in the cartesian coordinate space (to each
  coordinate independently), to correct for jitters in the trajectory caused by
  imperfections in the internal -> xyz optimization. This obviously is done
  after RMSD alignment.

License
-------
GPLv3