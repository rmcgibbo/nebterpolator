"""Determine cartesian coordinates from internal coordinates
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np
from scipy.optimize import leastsq

# local imports
from core import internal
from core import internal_derivs

##############################################################################
# Functions
##############################################################################


def least_squares_cartesian(bonds, ibonds, angles, iangles, dihedrals,
                            idihedrals, xyz_guess, **kwargs):
    """Determine a set of cartesian coordinates maximally-consistent with
    a set of redundant internal coordinates.

    This function operates on only a single frame at a time.

    It solves the overdetermined problem of finding a cartesian structure
    (3N-6 degrees of freedom) maximimally consistent with a set of more than
    3N-6 redundant internal coordinates by finding the cartesian structure
    than minimizes the least-squares deviation between the implied internal
    coordinates and the actual internal coordinates. We use the L-BFGS-B
    optimizer.

    Parameters
    ----------
    bonds : np.ndarray, shape=[n_bonds]
        The collected distances, such that distances[i] is the distance
        between the `i`th pair of atoms (whose indices are ibonds[i,0] and
        ibonds[i,1]).
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded.
    angles : np.ndarray, shape=[n_angles]
        The collected angles, such that angles[i] is the angle between
        the `i`th triplet of atoms (whose indices are iangles[i,0],
        iangles[i,1] and iangles[i,2]).
    iangles : np.ndarray, shape=[n_angles, 3], dtype=int
        Each row in `iangles` is a triplet of indicies `i`, `j`, `k`
        indicating that atoms `i` - `j` - `k` form an angle of interest.
    dihedrals : np.ndarray, shape=[n_dihedrals]
        The collected dihedrals, such that dihedrals[i] is the dihedral
        between the `i`th quartet of atoms (whose indices are idihedrals[i,0],
        idihedrals[i,1], idihedrals[i,2], and idihedrals[i,3]).
    idihedrals : np.ndarray, shape=[n_dihedrals, 4], dtype=int
        Each row in `idihedrals` is a quartet of indicies `i`, `j`, `k`, `l`,
        indicating that atoms `i` - `j` - `k` - `l` form a dihedral of
        interest.
    xyz_guess : np.ndarray, shape[n_atoms, 3]
        A guess for the cartesian coordinates. This will serve as a starting
        point for the optimization.

    Other Parameters
    ----------------
    approx_grad : bool, default=False
        Use numerical gradients instead of analytical gradients. This is
        slower, but may be useful for debugging.
    display : bool, default=True
        Display summary statistics from the L-BFGS-B optimizer to stdout
    """

    # TODO: expose the ability to set a weight vector over the different
    # internal coordinates that sets how they contribute to the objective
    # function. This might just be 3 elements -- (one for bonds, angles,
    # dihedrals). But it's a good idea since there are going to many more
    # dihedrals than bonds and angles, and the bond lengths are measured in
    # different units than the angles and dihedrals.

    display = kwargs.pop('display', True)
    approx_grad = kwargs.pop('approx_grad', False)
    for key in kwargs.keys():
        print '%s is not a recognized kwarg. ignored' % key

    if xyz_guess.ndim != 2:
        raise ValueError('cartesian_guess should be a 2d array')
    if len(bonds) != len(ibonds):
        raise ValueError('The size of bonds and ibonds doesn\'t match')
    if len(angles) != len(iangles):
        raise ValueError('The size of angles and iangles doesn\'t match')
    if len(dihedrals) != len(idihedrals):
        raise ValueError('The size of dihedrals and idihedrals doesn\'t match')

    n_atoms = xyz_guess.shape[0]
    reference_internal = np.concatenate([bonds, angles, dihedrals])

    def independent_vars_to_xyz(x):
        if x.ndim != 1:
            raise TypeError('independent variables must be 1d')
        if len(x) != 3*n_atoms - 6:
            raise TypeError('Must be 3N-6 independent variables')

        xyz = np.zeros((n_atoms, 3))

        # fill in 6 DOFs from the initial structure
        xyz[0, :] = xyz_guess[0, :]
        xyz[1, 0:2] = xyz_guess[1, 0:2]
        xyz[2, 0] = xyz_guess[2, 0]

        # the rest are independent variables
        xyz[1, 2] = x[0]
        xyz[2, 1] = x[1]
        xyz[2, 2] = x[2]
        xyz[3:, :] = x[3:].reshape(n_atoms-3, 3)

        return xyz

    def xyz_to_independent_vars(xyz):
        special_indices = [5, 7, 8]

        x = np.zeros(3*n_atoms - 6)
        flat = xyz.flatten()
        x[0:3] = flat[special_indices]
        x[3:] = flat[9:]

        return x

    def func(x):
        xyz = independent_vars_to_xyz(x)

        # these methods require 3d input
        xyzlist = np.array([xyz])
        bonds = internal.bonds(xyzlist, ibonds)
        angles = internal.angles(xyzlist, iangles)
        dihedrals = internal.dihedrals(xyzlist, idihedrals)

        # the internal coordinates corresponding to the current cartesian
        # 1-dimensional, of length n_internal
        current_internal = np.concatenate([bonds.flatten(), angles.flatten(),
                                           dihedrals.flatten()])
        result = current_internal - reference_internal

        if display:
            print 'SSD:', np.sum(np.square(result))
        return result

    def grad(x):
        xyz = independent_vars_to_xyz(x)

        d_bonds = internal_derivs.bond_derivs(xyz, ibonds)
        d_angles = internal_derivs.angle_derivs(xyz, iangles)
        d_dihedrals = internal_derivs.dihedral_derivs(xyz, idihedrals)

        # the derivatives of the internal coordinates wrt the cartesian
        # this is 2d, with shape equal to n_internal x n_cartesian
        d_internal = np.vstack([d_bonds.reshape((len(ibonds), -1)),
                                d_angles.reshape((len(iangles), -1)),
                                d_dihedrals.reshape((len(idihedrals), -1))])
        return d_internal

    x0 = xyz_to_independent_vars(xyz_guess)
    # make sure that we're extracting and reconstructing
    # the 3N-6 correctly
    np.testing.assert_equal(independent_vars_to_xyz(x0), xyz_guess)

    if approx_grad:
        print 'approx grad'
        x, cov_x, info, msg, iflag = leastsq(func, full_output=True, x0=x0)
    else:
        x, cov_x, info, msg, iflag = leastsq(func, col_deriv=grad,
                                             full_output=True, x0=x0, ftol=1e-10)

    xyz_final = independent_vars_to_xyz(x)
    if display:
        print 'FINAL SSD:', np.sum(np.square(info['fvec']))
    if not iflag in [1, 2, 3, 4]:
        # these are the sucess values if the flag
        raise Exception(msg)
    return xyz_final


def main():
    from path_operations import union_connectivity
    #np.random.seed(42)
    xyzlist = 0.1*np.random.randn(7, 5, 3)
    atom_names = ['C' for i in range(5)]

    ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names)

    bonds = internal.bonds(xyzlist, ibonds)
    angles = internal.angles(xyzlist, iangles)
    dihedrals = internal.dihedrals(xyzlist, idihedrals)

    xyz_guess = xyzlist[0] + 0.025*np.random.rand(*xyzlist[0].shape)
    x = least_squares_cartesian(bonds[0], ibonds, angles[0], iangles,
                                dihedrals[0], idihedrals, xyz_guess)

    print x
    #print xyzlist[0]

if __name__ == '__main__':
    main()
