"""
Routines to etermine the bond/angle/dihedral connectivity in a molecular graph.
"""

##############################################################################
# Imports
##############################################################################

import numpy as np
from scipy.spatial.distance import squareform, pdist
import networkx as nx
from itertools import combinations, ifilter

##############################################################################
# GLOBALS
##############################################################################

# these are covalent radii taken from the crystalographic data in nm
# Dalton Trans., 2008, 2832-2838, DOI: 10.1039/B801115J
# http://pubs.rsc.org/en/Content/ArticleLanding/2008/DT/b801115j
COVALENT_RADII = {'C': 0.0762, 'N': 0.0706, 'O': 0.0661, 'H': 0.031,
                  'S': 0.105}
              
__all__ = ['bond_connectivity', 'angle_connectivity', 'dihedral_connectivity']

##############################################################################
# Functions
##############################################################################


def bond_connectivity(xyz, atom_names):
    """Get a list of all the bonds in a conformation

    Regular bonds are assigned to all pairs of atoms where
    the interatomic distance is less than or equal to 1.3 times the
    sum of their respective covalent radii.

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation. The coodinates
        are expected to be in units of nanometers.
    atom_names : array_like of strings, length=n_atoms
        A list of the names of each of the atoms, which will be used for
        grabbing the covalent radii.

    Returns
    -------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        n_bonds x 2 array of indices, where each row is the index of two
        atom who participate in a bond.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    xyz = np.asarray(xyz)
    if not xyz.ndim == 2:
        raise TypeError('xyz has ndim=%d. Should be 2' % xyz.ndim)

    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    if len(atom_names) != n_atoms:
        raise ValueError(('atom_names must have the same number of atoms '
                          'as xyz'))

    # TODO: This logic only works for elements that are a single letter
    # If we need to deal with other elements, we can easily generalize it.
    proper_atom_names = np.zeros(n_atoms, dtype='S1')
    for i in xrange(n_atoms):
        # name of the element that is atom[i]
        # take the first character of the AtomNames string,
        # after stripping off any digits
        proper_atom_names[i] = atom_names[i].strip('123456789 ')[0]
        if not proper_atom_names[i] in COVALENT_RADII.keys():
            raise ValueError("I don't know about this atom_name: %s" %
                             atom_names[i])

    distance_mtx = squareform(pdist(xyz))
    connectivity = []

    for i in xrange(n_atoms):
        for j in xrange(i+1, n_atoms):
            # Regular bonds are assigned to all pairs of atoms where
            # the interatomic distance is less than or equal to 1.3 times the
            # sum of their respective covalent radii.
            d = distance_mtx[i, j]
            if d < 1.3 * (COVALENT_RADII[proper_atom_names[i]] +
                          COVALENT_RADII[proper_atom_names[j]]):
                connectivity.append((i, j))

    return np.array(connectivity)


def angle_connectivity(ibonds):
    """Given the bonds, get the indices of the atoms defining all the bond
    angles

    A 'bond angle' is defined as any set of 3 atoms, `i`, `j`, `k` such that
    atom `i` is bonded to `j` and `j` is bonded to `k`

    Parameters
    ----------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    iangles : np.ndarray, shape[n_angles, 3], dtype=int
        n_angles x 3 array of indices, where each row is the index of three
        atoms m,n,o such that n is bonded to both m and o.
    """

    graph = nx.from_edgelist(ibonds)
    iangles = []

    for i in graph.nodes():
        for (m, n) in combinations(graph.neighbors(i), 2):
            # so now the there is a bond angle m-i-n
            iangles.append((m, i, n))

    return np.array(iangles)


def dihedral_connectivity(ibonds):
    """Given the bonds, get the indices of the atoms defining all the dihedral
    angles

    A 'dihedral angle' is defined as any set of 4 atoms, `i`, `j`, `k`, `l`
    such that atom `i` is bonded to `j`, `j` is bonded to `k`, and `k` is
    bonded to `l`.

    Parameters
    ----------
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    idihedrals : np.ndarray, shape[n_dihedrals, 4], dtype=int
        All sets of 4 atoms `i`, `j`, `k`, `l` such that such that atom `i` is
        bonded to `j`, `j` is bonded to `k`, and `k` is bonded to `l`.
    """
    graph = nx.from_edgelist(ibonds)
    idihedrals = []

    # TODO: CHECK FOR DIHEDRAL ANGLES THAT ARE 180 and recover
    # conf : msmbuilder.Trajectory
    #    An msmbuilder trajectory, only the first frame will be used. This
    #    is used purely to make the check for angle(ABC) != 180.

    for a in graph.nodes():
        for b in graph.neighbors(a):                
            for c in ifilter(lambda c: c not in [a, b], graph.neighbors(b)):
                for d in ifilter(lambda d: d not in [a, b, c], graph.neighbors(c)):
                    idihedrals.append((a, b, c, d))

    return np.array(idihedrals)
