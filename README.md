nebterpolator
=============

A library for generating smooth paths from ab-inito MD trajectories. These
paths might serve as input for an NEB calculation. The key component of this
package is the ability to translate from cartesian coordinates to redundant
internal coordinates (bond lengths, angles, dihedrals) and to go the other
way, from redundant internal coordinates to an "optimal" set of cartesian
coordinates.

Pipeline
---------

- Select a set of redundant internal coordinates
  - currently, we're using ALL pairwise distances, all of the angles between
    sets of three atoms, a-b-c, that actually get "bonded" during the
    trajectory, and all of the dihedral angles between sets of 4 atoms,
    a-b-c-d, that actually get "bonded" during the trajectory. I observed
    empirically that using ALL pairwise distances seems to work better than
    only using pairs corresponding to actually bonded atoms.
- Transform each frame in a trajectory into the internal coordinate space
- Apply smoothing filter to the time-series from each internal coordinate
  degree of freedom independently. Currently, we're using a zero-delay
  low pass Buttersworth filter. A digital convolution filter and a polynomial
  fit smoother are also available in the package though.
   - The cutoff frequency is _the_ key variable for the algorithm.
- For each frame, find a set of xyz coordinates that are optimally consistent
  with the smoothed internal coordinates.
   - This is a computationally intensive optimization problem, even with
     analytic derivatives and Levenberg-Marquardt. Fortunately it's embarrassingly
     parallel over the frames in the trajectory, as each frame is optimized
     separately. The code is currently running this step across multiple nodes
     with MPI.
- Apply a very slight smoothing in the cartesian coordinate space, to correct
  for jitters in the trajectory caused by imperfections in the internal -> xyz
  optimization. This obviously is done after RMSD alignment.


Code in the library
-------------------
There's some code in this library that might be of interest for other
tasks, besides trajectory smoothing.

- Calculation of bond lengths, bond angles and dihedral angles from cartesian
  coordinates.
- Analytic derivatives of bond lengths, bond angles, and dihedral angles with
  respect to cartesian coordinates.
- Find a set of cartesian coordinates given redundant internal coordinates
  as input, via a least-squares optimization (Levenberg-Marquardt).
- Context managers that make MPI code in python much more elegant.
- Kabsch RMSD in numpy.

Requirements
------------
- python 2.6+  (tested with 2.7.3)
- numpy, 1.6+  (tested with 1.6.2)
- scipy, 0.10+ (tested with 0.10.1)
- mpi4py 1.3+  (tested with 1.3)

Getting Started
---------------

Open up the file `client.py` and take a look. Try running it, either serially
or with mpi

```
$ python client.py
$ mpirun -np 2 client.py
```

License
-------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.