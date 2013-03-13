"""Methods for interpolating a smooth path in internal coordinates.
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

# local imports
from core import internal
from core import internal_derivs

##############################################################################
# Functions
##############################################################################

def least_squares_cartesian(bonds, ibonds, angles, iangles, dihedrals,
                            idihedrals, cartesian_guess, **kwargs):
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
    cartesian_guess : np.ndarray, shape[n_atoms, 3]
        A guess for the cartesian coordinates. This will serve as a starting
        point for the optimization.

    Other Parameters
    ----------------
    approx_grad : bool, default=False
        Use numerical gradients instead of analytical gradients. This is
        slower, but may be useful for debugging.
    display : bool, default=True
        Display summary statistics from the L-BFGS-B optimizer to stdout
    m : int, default=20
        The maximum number of variable metric corrections used to define the
        limited memory matrix. (The limited memory BFGS method does not store
        the full hessian but uses this many terms in an approximation to it.)
    """

    # TODO: Remove 6 degrees of freedom from the cartesian optimization, to
    # exploit the fact that we already KNOW that there is this 6 dimensional
    # manifold on which the objective function is constant.

    # TODO: expose the ability to set a weight vector over the different
    # internal coordinates that sets how they contribute to the objective
    # function. This might just be 3 elements -- (one for bonds, angles,
    # dihedrals). But it's a good idea since there are going to many more
    # dihedrals than bonds and angles, and the bond lengths are measured in
    # different units than the angles and dihedrals.

    approx_grad = kwargs.pop('approx_grad', False)
    display = kwargs.pop('display', True)
    m = kwargs.pop('m', 20)

    if cartesian_guess.ndim != 2:
        raise ValueError('cartesian_guess should be a 2d array')
    if len(bonds) != len(ibonds):
        raise ValueError('The size of bonds and ibonds doesn\'t match')
    if len(angles) != len(iangles):
        raise ValueError('The size of angles and iangles doesn\'t match')
    if len(dihedrals) != len(idihedrals):
        raise ValueError('The size of dihedrals and idihedrals doesn\'t match')

    cartesian_shape = cartesian_guess.shape
    reference_internal = np.concatenate([bonds, angles, dihedrals])

    def objective(flat_cartesian):
        if flat_cartesian.ndim != 1:
            raise TypeError('objective takes flattened cartesian coordinates')

        x = flat_cartesian.reshape(cartesian_shape)

        bonds = internal.bonds([x], ibonds)
        angles = internal.angles([x], iangles)
        dihedrals = internal.dihedrals([x], idihedrals)

        # the internal coordinates corresponding to the current cartesian
        # 1-dimensional, of length n_internal
        current_internal = np.concatenate([bonds.flatten(), angles.flatten(),
                                           dihedrals.flatten()])
        error = 0.5 * np.sum((current_internal - reference_internal)**2)

        if approx_grad:
            # if we're using the approx_grad, we don't want to actually return
            # the grads
            return error

        d_bonds = internal_derivs.bond_derivs(x, ibonds)
        d_angles = internal_derivs.angle_derivs(x, iangles)
        d_dihedrals = internal_derivs.dihedral_derivs(x, idihedrals)

        # the derivatives of the internal coordinates wrt the cartesian
        # this is 2d, with shape equal to n_internal x n_cartesian
        d_internal = np.vstack([d_bonds.reshape((len(ibonds), -1)),
                                d_angles.reshape((len(iangles), -1)),
                                d_dihedrals.reshape((len(idihedrals), -1))])

        grad_error = np.dot((current_internal - reference_internal),
                            d_internal)

        return error, grad_error

    iprint = 0 if display else -1
    x, f, d = fmin_l_bfgs_b(objective, cartesian_guess.flatten(),
                            iprint=iprint, approx_grad=approx_grad, m=m)

    final_cartesian = x.reshape(cartesian_shape)

    if d['warnflag'] != 0:
        raise ValueError('The optimization did not converge.')

    return final_cartesian


def main():
    from path_operations import union_connectivity
    np.random.seed(10)
    xyzlist = 0.1*np.random.randn(7, 5, 3)
    atom_names = ['C' for i in range(5)]

    ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names)

    bonds = internal.bonds(xyzlist, ibonds)
    angles = internal.angles(xyzlist, iangles)
    dihedrals = internal.dihedrals(xyzlist, idihedrals)

    x = least_squares_cartesian(bonds[0], ibonds, angles[0], iangles,
                                dihedrals[0], idihedrals, xyzlist[1], m=20)

    print x
    print xyzlist[0]

if __name__ == '__main__':
    main()
