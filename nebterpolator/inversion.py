"""Determine cartesian coordinates from internal coordinates
"""

##############################################################################
# Imports
##############################################################################

# library imports
import numpy as np
import sys
from scipy.optimize import leastsq, fmin, fmin_l_bfgs_b
try:
    from nanoreactor.molecule import Molecule
    from nanoreactor.morse_function import PairwiseMorse
    have_molecule = 1
except:
    have_molecule = 0

# local imports
import core

##############################################################################
# Globals
##############################################################################

__all__ = ['least_squares_cartesian']

##############################################################################
# Functions
##############################################################################

def least_squares_cartesian(bonds, ibonds, angles, iangles, dihedrals,
                                  idihedrals, xyz_guess, elem, w_morse, **kwargs):
    """Determine a set of cartesian coordinates maximally-consistent with
    a set of redundant internal coordinates.

    Additionally, add a term corresponding to a Morse potential
    between all pairs of atoms to minimize the number of unphysical
    structures.

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
    verbose : bool, default=True
        Display summary statistics from the L-BFGS-B optimizer to stdout
    xref : np.ndarray, shape=[n_atoms, 3]
        Another set of XYZ coordinates to act as an anchor
    w_xref : float
        The weight of the anchor coordinates
    w_morse : float
        The weight of the Morse potential
    elem : list
        Names of the elements used in constructing the Morse potential

    Returns
    -------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The optimized xyz coordinates
    error : float
        The RMS deviation across internal DOFs
    """

    if not have_molecule:
        raise ImportError('The least_squares_cartesian function requires the nanoreactor.molecule module')

    # TODO: expose the ability to set a weight vector over the different
    # internal coordinates that sets how they contribute to the objective
    # function. This might just be 3 elements -- (one for bonds, angles,
    # dihedrals). But it's a good idea since there are going to many more
    # dihedrals than bonds and angles, and the bond lengths are measured in
    # different units than the angles and dihedrals.

    verbose = kwargs.pop('verbose', False)
    xref = kwargs.pop('xref', None)
    w_xref = kwargs.pop('w_xref', 1.0)
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

    def gxyz_to_independent_vars(grad):
        special_indices = [5, 7, 8]
        g = np.zeros((grad.shape[0], 3*n_atoms - 6), dtype=float)
        g[:, 0:3] = grad[:, special_indices]
        g[:, 3:] = grad[:, 9:]

        return g

    if xref != None:
        xrefi = xyz_to_independent_vars(xref)
    else:
        xrefi = None

    wdeg = (180.0 / np.pi) / 30 # Number of degrees in a radian, divided by our somewhat arbitrary choice of what constitutes a "major" deviation in angle.
    w1 = 10.0 # Angstrom is a somewhat natural unit of bonds.
    w2 = wdeg * np.sqrt(len(bonds)) / np.sqrt(len(angles)) # Keep radian but normalize to the number of bonds.
    w3 = wdeg * np.sqrt(len(bonds)) / np.sqrt(len(dihedrals)) # Keep radian but normalize to the number of bonds.

    def fgrad(x, indicate = False):
        """ Calculate the objective function and its derivatives. """
        # If the optimization algorithm tries to calculate twice for the same point, do nothing.
        # if x == fgrad.x0: return

        xyz = independent_vars_to_xyz(x)
        # these methods require 3d input
        xyzlist = np.array([xyz])
        my_bonds = core.bonds(xyzlist, ibonds).flatten()
        my_angles = core.angles(xyzlist, iangles).flatten()
        my_dihedrals = core.dihedrals(xyzlist, idihedrals, anchor=dihedrals).flatten()

        # Deviations of internal coordinates from ideal values.
        d1 = w1*(my_bonds - bonds)
        d2 = w2*(my_angles - angles)
        d3 = w3*(my_dihedrals - dihedrals)

        # Include an optional term if we have an anchor point.
        if xrefi != None:
            d4 = (x - xrefi).flatten() * w1 * w_xref
            fgrad.error = np.r_[d1, d2, np.arctan2(np.sin(d3), np.cos(d3)), d4]
        else:
            fgrad.error = np.r_[d1, d2, np.arctan2(np.sin(d3), np.cos(d3))]

        # The objective function contains another contribution from the Morse potential.
        fgrad.X = np.dot(fgrad.error, fgrad.error)
        d1s = np.dot(d1, d1)
        d2s = np.dot(d2, d2)
        d3s = np.dot(d3, d3)
        M = Molecule()
        M.elem = elem
        M.xyzs = [np.array(xyz)*10]
        if w_morse != 0.0:
            EMorse, GMorse = PairwiseMorse(M)
            EMorse = EMorse[0]
            GMorse = GMorse[0]
        else:
            EMorse = 0.0
            GMorse = np.zeros((n_atoms, 3), dtype=float)
        if indicate: 
            if fgrad.X0 != None:
                print ("LSq: %.4f (%+.4f) Distance: %.4f (%+.4f) Angle: %.4f (%+.4f) Dihedral: %.4f (%+.4f) Morse: % .4f (%+.4f)" % 
                       (fgrad.X, fgrad.X - fgrad.X0, d1s, d1s - fgrad.d1s0, d2s, d2s - fgrad.d2s0, d3s, d3s - fgrad.d3s0, EMorse, EMorse - fgrad.EM0)), 
            else:
                print "LSq: %.4f Distance: %.4f Angle: %.4f Dihedral: %.4f Morse: % .4f" % (fgrad.X, d1s, d2s, d3s, EMorse), 
                fgrad.X0 = fgrad.X
                fgrad.d1s0 = d1s
                fgrad.d2s0 = d2s
                fgrad.d3s0 = d3s
                fgrad.EM0 = EMorse
        fgrad.X += w_morse*EMorse

        # Derivatives of internal coordinates w/r.t. Cartesian coordinates.
        d_bonds = core.bond_derivs(xyz, ibonds) * w1
        d_angles = core.angle_derivs(xyz, iangles) * w2
        d_dihedrals = core.dihedral_derivs(xyz, idihedrals) * w3

        if xrefi != None:
            # the derivatives of the internal coordinates wrt the cartesian
            # this is 2d, with shape equal to n_internal x n_cartesian
            d_internal = np.vstack([gxyz_to_independent_vars(d_bonds.reshape((len(ibonds), -1))),
                                    gxyz_to_independent_vars(d_angles.reshape((len(iangles), -1))),
                                    gxyz_to_independent_vars(d_dihedrals.reshape((len(idihedrals), -1))),
                                    np.eye(len(x)) * w1 * w_xref])
        else:
            # the derivatives of the internal coordinates wrt the cartesian
            # this is 2d, with shape equal to n_internal x n_cartesian
            d_internal = np.vstack([gxyz_to_independent_vars(d_bonds.reshape((len(ibonds), -1))),
                                    gxyz_to_independent_vars(d_angles.reshape((len(iangles), -1))),
                                    gxyz_to_independent_vars(d_dihedrals.reshape((len(idihedrals), -1)))])

        # print fgrad.error.shape, d_internal.shape
        # print d_internal.shape
        # print xyz_to_independent_vars(d_internal).shape
        fgrad.G = 2*np.dot(fgrad.error, d_internal)
        fgrad.G += xyz_to_independent_vars(w_morse*GMorse.flatten())
    fgrad.do_G = False
    fgrad.x0 = None
    fgrad.error = None
    fgrad.X = None
    fgrad.G = None
    fgrad.X0 = None
    fgrad.EM0 = None

    def func(x, indicate = False):
        fgrad(x, indicate = indicate)
        return fgrad.X

    def grad(x):
        fgrad.do_G = True
        fgrad(x)
        return fgrad.G

    x0 = xyz_to_independent_vars(xyz_guess)
    # make sure that we're extracting and reconstructing
    # the 3N-6 correctly
    np.testing.assert_equal(independent_vars_to_xyz(x0), xyz_guess)

    #====
    # Finite difference code to check that gradients are correct!
    #====
    def fdwrap(x_, idx):
        def func1(arg):
            x1 = x_.copy()
            x1[idx] += arg
            return func(x1)
        return func1

    def f1d7p(f, h):
        """
        A highly accurate seven-point finite difference stencil
        for computing derivatives of a function.  
        """
        fm3, fm2, fm1, f1, f2, f3 = [f(i*h) for i in [-3, -2, -1, 1, 2, 3]]
        fp = (f3-9*f2+45*f1-45*fm1+9*fm2-fm3)/(60*h)
        return fp

    FDCheck = False
    if FDCheck:
        AGrad = grad(x0)
        FDGrad = np.array([f1d7p(fdwrap(x0, i), h=0.001) for i in range(len(x0))])
        print "Analytic Gradient (Initial):", AGrad
        print "Error in Gradient:", FDGrad - AGrad
        raw_input()
    #====
    # End finite difference code
    #====
        
    print "Initial:", 
    func(x0, indicate = True)
    print
    # try:
    xf, ff, d = fmin_l_bfgs_b(func,x0,fprime=grad,m=30,factr=1e7,pgtol=1e-4,iprint=-1,disp=0,maxfun=1e5,maxiter=1e5)
    print "Final:", 
    func(xf, indicate = True)
    print
    if FDCheck:
        AGrad = grad(xf)
        FDGrad = np.array([f1d7p(fdwrap(xf, i), h=0.001) for i in range(len(xf))])
        print "Analytic Gradient (Final):", AGrad
        print "Error in Gradient:", FDGrad - AGrad
        raw_input()
    xyz_final = independent_vars_to_xyz(xf)
    # except:
    #     raise RuntimeError
    #     xyz_final = xyz_guess
    #     ff = 1e10

    return xyz_final, ff


def main():
    from path_operations import union_connectivity
    #np.random.seed(42)
    xyzlist = 0.1*np.random.randn(7, 5, 3)
    atom_names = ['C' for i in range(5)]

    ibonds, iangles, idihedrals = union_connectivity(xyzlist, atom_names)

    bonds = core.bonds(xyzlist, ibonds)
    angles = core.angles(xyzlist, iangles)
    dihedrals = core.dihedrals(xyzlist, idihedrals)

    xyz_guess = xyzlist[0] + 0.025*np.random.rand(*xyzlist[0].shape)
    x = least_squares_cartesian(bonds[0], ibonds, angles[0], iangles,
                                dihedrals[0], idihedrals, xyz_guess)

    print x
    #print xyzlist[0]

if __name__ == '__main__':
    main()
