"""
Numpy routines to calculate the analytical second derivatibes of internal
coordinates (bonds, angles, dihedrals) with respect to cartesian coordinates
"""

##############################################################################
# Imports
##############################################################################

import numpy as np

import internal
import internal_derivs

##############################################################################
# GLOBALS
##############################################################################

VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)

__all__ = ['bond_hessian', 'angle_heddian', 'dihedral_hessian']

##############################################################################
# Functions
##############################################################################


def bond_hessian(xyz, ibonds):
    """
    Hessian of the bond lengths with respect to cartesian coordinates

    Parameters
    ----------
    xyz : np.ndarray, shape=[n_atoms, 3]
        The cartesian coordinates of a single conformation.
    ibonds : np.ndarray, shape=[n_bonds, 2], dtype=int
        Each row in `ibonds` is a pair of indicies `i`, `j`, indicating that
        atoms `i` and `j` are bonded

    Returns
    -------
    bond_hessian : np.ndarray, shape=[n_bonds, n_atoms, 3, n_atoms, 3]
        The hessian is a 5d array, where bond_derivs[i,l,m,n,o] gives the
        derivative of the `i`th bond length (the bond between atoms
        ibonds[i,0] and ibonds[i,1]) with respect to the `l`th atom's
        `m`th coordinate and the `n`th atom's `o`th coordinate.

    References
    ----------
    Bakken and Helgaker, JCP Vol. 117, Num. 20 22 Nov. 2002
    http://folk.uio.no/helgaker/reprints/2002/JCP117b_GeoOpt.pdf
    """
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_bonds, two = ibonds.shape
    if two != 2:
        raise TypeError('ibonds must have 2 columns.')
    
    hessian = np.zeros((n_bonds, n_atoms, 3, n_atoms, 3))
    for b, (m, n) in enumerate(ibonds):
        u_prime = (xyz[m] - xyz[n])
        length = np.linalg.norm(u_prime)
        u = u_prime / length
        term = (np.outer(u, u) - np.eye(3)) / length

        hessian[b, m, :, n, :] = term
        hessian[b, m, :, m, :] = -term
        hessian[b, n, :, m, :] = term
        hessian[b, n, :, n, :] = -term
    
    return hessian


def angle_hessian(xyz, iangles):
    n_atoms, three = xyz.shape
    if three != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_angles, three = iangles.shape
    if three != 3:
        raise TypeError('angles must have 3 columns.')
    
    qa = internal.angles(xyz.reshape(1, n_atoms, 3), iangles).flatten()
    jacobian = internal_derivs.angle_derivs(xyz, iangles)
    
    hessian = np.zeros((n_angles, n_atoms, 3, n_atoms, 3))
    
    def sign(i,j,k):
        if i == j:
            return 1
        if i == k:
            return - 1
        else:
            return 0
        
    for i, (m, o, n) in enumerate(iangles):
        u_prime = xyz[m] - xyz[o]
        v_prime = xyz[n] - xyz[o]
        lambda_u = np.linalg.norm(u_prime)
        lambda_v = np.linalg.norm(v_prime)
        u = u_prime / lambda_u
        v = v_prime / lambda_v
        jac = jacobian[i]

        cos = np.cos(qa[i])
        sin = np.sin(qa[i])
        uv = np.outer(u, v)
        uu = np.outer(u, u)
        vv = np.outer(v, v)
        eye = np.eye(3)
        
        term1 = (uv + uv.T + (-3 * uu + eye) * cos) / (lambda_u**2 * sin)
        term2 = (uv + uv.T + (-3 * vv + eye) * cos) / (lambda_v**2 * sin)
        term3 = (uu + vv - uv   * cos - eye) / (lambda_u * lambda_v * sin)
        term4 = (uu + vv - uv.T * cos - eye) / (lambda_u * lambda_v * sin)
        hessian[i] = -(cos / sin) * np.outer(jac.flatten(), jac.flatten()).reshape(n_atoms, 3, n_atoms, 3)
        
        for a in [m, n, o]:
            for b in [m, n, o]:
                hessian[i, a, :, b, :] += (sign(a,m,o)*sign(b,m,o) * term1)
                hessian[i, a, :, b, :] += (sign(a,n,o)*sign(b,n,o) * term2)
                hessian[i, a, :, b, :] += (sign(a,m,o)*sign(b,n,o) * term3)
                hessian[i, a, :, b, :] += (sign(a,n,o)*sign(b,m,o) * term4)
    
    return hessian


if __name__ == '__main__':
    import internal_derivs
    
    h = 1e-7
    xyz = np.random.randn(4,3)
    xyz2 = xyz.copy()
    xyz2[1,1] += h
    ibonds = np.array([[0,1], [0,2]])
    iangles = np.array([[0,1,2], [1,2,3]])

    print 'TESTING BOND HESSIAN'
    jac1 = internal_derivs.bond_derivs(xyz, ibonds)
    jac2 = internal_derivs.bond_derivs(xyz2, ibonds)
    hessian = bond_hessian(xyz, ibonds)
    print ((jac2-jac1)/h)[0]
    print hessian[0, 1, 1]
    
    
    print '\nTESTING ANGLE HESSIAN'
    jac1 = internal_derivs.angle_derivs(xyz, iangles)
    jac2 = internal_derivs.angle_derivs(xyz2, iangles)
    hessian = angle_hessian(xyz, iangles)
    print ((jac2-jac1)/h)[0]
    print hessian[0, 1, 1] 
    
