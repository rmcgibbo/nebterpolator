#cython: boundscheck=False
#cython: wraparound=False
from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, acos, cos, sin, atan2


def bond_lengths(np.ndarray[np.double_t, ndim=2, mode='c'] xyz not None,
         np.ndarray[np.int_t, ndim=2, mode='c'] ibonds not None,
         int deriv=0):
    """Calculate the bond lengths (and derivatives)
    """
    cdef int i, m, n, n_atoms, n_bonds
    cdef double norm
    cdef int j, k
    cdef double t01, t02, t12
    cdef np.ndarray[ndim=2, dtype=np.double_t] hessian_term0
    cdef np.ndarray[ndim=1, dtype=np.double_t] u

    # do some checking
    n_atoms = xyz.shape[0]
    if xyz.shape[1] != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_bonds = ibonds.shape[0]
    if ibonds.shape[1] != 2:
        raise TypeError('ibonds must have 2 columns.')
    if not ((deriv == 0) or (deriv == 1) or (deriv == 2)):
        raise ValueError('deriv must be 0, 1, 2')

    # declare the output arrays
    bond_lengths = np.zeros(n_bonds, dtype=np.double)
    if deriv >= 1:
        jacobian = np.zeros((n_bonds, n_atoms, 3), dtype=np.double)
    if deriv == 2:
        hessian = np.zeros((n_bonds, n_atoms, 3, n_atoms, 3), dtype=np.double)

    for i in range(n_bonds):
        m = ibonds[i,0]
        n = ibonds[i,1]

        u = xyz[m] - xyz[n]
        norm = norm3(u)
        bond_lengths[i] = norm

        if deriv >= 1:
            u = u / norm
            jacobian[i, m, :] = u
            jacobian[i, n, :] = -u

        if deriv == 2:
            hessian_term0 = outer3_minus_eye(u, u) / norm
            hessian[i, m, :, n, :] = hessian_term0
            hessian[i, m, :, m, :] = -hessian_term0
            hessian[i, n, :, m, :] = hessian_term0
            hessian[i, n, :, n, :] = -hessian_term0

    if deriv == 0:
        return bond_lengths
    elif deriv == 1:
        return bond_lengths, jacobian
    else:
        return bond_lengths, jacobian, hessian


def bond_angles(np.ndarray[np.double_t, ndim=2, mode='c'] xyz not None,
         np.ndarray[np.int_t, ndim=2, mode='c'] iangles not None,
         int deriv=0):
    """Calculate the bond angles (and derivatives)
    """
    cdef int i, m, o, n, n_atoms, n_angles, a, b
    cdef double u_norm, v_norm, angle, cos_angle, sin_angle
    cdef np.ndarray[ndim=1, dtype=np.double_t] u, u_prime
    cdef np.ndarray[ndim=1, dtype=np.double_t] v, v_prime,
    cdef np.ndarray[ndim=1, dtype=np.double_t] w, w_prime
    cdef np.ndarray[ndim=1, dtype=np.double_t] jacobian_term0, jacobian_term1
    cdef np.ndarray[ndim=2, dtype=np.double_t] uu, vv, uv, term1, term2, term3, term4

    n_atoms = xyz.shape[0]
    if xyz.shape[1] != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_angles = iangles.shape[0]
    if iangles.shape[1] != 3:
        raise TypeError('iangles must have 3 columns.')
    if not ((deriv == 0) or (deriv == 1) or (deriv == 2)):
        raise ValueError('deriv must be 0, 1, 2')

    # declare the output arrays
    bond_angles = np.zeros(n_angles, dtype=np.double)
    if deriv >= 1:
        jacobian = np.zeros((n_angles, n_atoms, 3), dtype=np.double)
    if deriv == 2:
        hessian = np.zeros((n_angles, n_atoms, 3, n_atoms, 3), dtype=np.double)

    for i in range(n_angles):
        m = iangles[i, 0]
        o = iangles[i, 1]
        n = iangles[i, 2]

        u_prime = xyz[m] - xyz[o]
        v_prime = xyz[n] - xyz[o]
        u_norm = norm3(u_prime)
        v_norm = norm3(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm

        angle = acos(dot3(u_prime, v_prime) / (u_norm*v_norm))
        bond_angles[i] = angle

        if deriv >= 1:
            if norm3(u+v) < 1e-10 or norm3(u-v) < 1e-10:
                raise ValueError('Sorry. I\'ll get back to this')
            else:
                w_prime = cross3(u, v)

            w  = w_prime / norm3(w_prime)
            jacobian_term0 = cross3(u, w) / u_norm
            jacobian_term1 = cross3(w, v) / v_norm

            jacobian[i, m, :] = jacobian_term0
            jacobian[i, n, :] = jacobian_term1
            jacobian[i, o, :] = -(jacobian_term0 + jacobian_term1)


        if deriv == 2:
            cos_angle = cos(angle)
            sin_angle = sin(angle)
            uv = outer3(u, v)
            uu = outer3(u, u)
            vv = outer3(v, v)
            eye = np.eye(3)

            term1 = (uv + uv.T + (-3 * uu + eye) * cos_angle) / (u_norm**2 * sin_angle)
            term2 = (uv + uv.T + (-3 * vv + eye) * cos_angle) / (v_norm**2 * sin_angle)
            term3 = (uu + vv - uv   * cos_angle - eye) / (u_norm * v_norm * sin_angle)
            term4 = (uu + vv - uv.T * cos_angle - eye) / (u_norm * v_norm * sin_angle)
            hessian[i] = -(cos_angle / sin_angle) * \
                np.outer(jacobian[i], jacobian[i]).reshape(n_atoms, 3, n_atoms, 3)

            for a in [m, n, o]:
                for b in [m, n, o]:
                    if sign6(a,m,o, b,m,o) != 0:
                        hessian[i, a, :, b, :] += sign6(a,m,o, b,m,o) * term1
                    if sign6(a,n,o, b,n,o) != 0:
                        hessian[i, a, :, b, :] += sign6(a,n,o, b,n,o) * term2
                    if sign6(a,m,o, b,n,o) != 0:
                        hessian[i, a, :, b, :] += sign6(a,m,o, b,n,o) * term3
                    if sign6(a,n,o, b,m,o) != 0:
                        hessian[i, a, :, b, :] += sign6(a,n,o, b,m,o) * term4

    if deriv == 0:
        return bond_angles
    elif deriv == 1:
        return bond_angles, jacobian
    else:
        return bond_angles, jacobian, hessian


def dihedral_angles(np.ndarray[np.double_t, ndim=2, mode='c'] xyz not None,
                    np.ndarray[np.int_t, ndim=2, mode='c'] idihedrals not None,
                    int deriv=0):
    """Calculate the dihedral angles (and derivatives)
    """
    cdef int i, m, o, p, n, n_atoms, n_dihedrals
    cdef double u_norm, v_norm, w_norm
    cdef np.ndarray[ndim=1, dtype=np.double_t] u, u_prime
    cdef np.ndarray[ndim=1, dtype=np.double_t] v, v_prime,
    cdef np.ndarray[ndim=1, dtype=np.double_t] w, w_prime
    cdef np.ndarray[ndim=1, dtype=np.double_t] cross_vw, cross_uw
    cdef np.ndarray[ndim=1, dtype=np.double_t] term1, term2, term3, term4

    n_atoms = xyz.shape[0]
    if xyz.shape[1] != 3:
        raise TypeError('xyz must be of length 3 in the last dimension.')
    n_dihedrals = idihedrals.shape[0]
    if idihedrals.shape[1] != 4:
        raise TypeError('idihedrals must have 4 columns.')
    if not ((deriv == 0) or (deriv == 1) or (deriv == 2)):
        raise ValueError('deriv must be 0, 1, 2')

    # declare the output arrays
    dihedral_angles = np.zeros(n_dihedrals, dtype=np.double)
    if deriv >= 1:
        jacobian = np.zeros((n_dihedrals, n_atoms, 3), dtype=np.double)
    if deriv == 2:
        hessian = np.zeros((n_dihedrals, n_atoms, 3, n_atoms, 3), dtype=np.double)

    for i in range(n_dihedrals):
        m = idihedrals[i, 0]
        o = idihedrals[i, 1]
        p = idihedrals[i, 2]
        n = idihedrals[i, 3]
        
        u_prime = xyz[m] - xyz[o]
        v_prime = xyz[n] - xyz[p]
        w_prime = xyz[p] - xyz[o]
        
        u_norm = norm3(u_prime)
        v_norm = norm3(v_prime)
        w_norm = norm3(w_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        w = w_prime / w_norm
        dot_uw = dot3(u, w)
        dot_vw = dot3(v, w)
        
        # use the non-normalized vectors
        cross_vw = cross3(v_prime, w_prime)
        cross_uw = cross3(u_prime, w_prime)
        dihedral_angles[i] = atan2(dot3(u_prime, cross_vw) * w_norm, dot3(cross_vw, cross_uw))
        # dihedral_angles[i] = acos(dot3(cross3(u, w), cross3(v, w)) / sqrt((1-dot_uw*dot_uw) * (1-dot_wv*dot_wv)))
        
        if deriv >= 1:
            # now normalize the vectors
            cross_vw /= (v_norm * w_norm)
            cross_uw /= (u_norm * w_norm)
            
            term1 = cross_uw / (u_norm * (1 - dot_uw*dot_uw))
            term2 = cross_vw / (v_norm * (1 - dot_vw*dot_vw))
            term3 = cross_uw * dot_uw / (w_norm * (1 - dot_uw*dot_uw))
            term4 = -cross_vw * dot_vw / (w_norm * (1 - dot_vw*dot_vw))

            jacobian[i, m, :] = term1
            jacobian[i, n, :] = -term2
            jacobian[i, o, :] = -term1 + term3 - term4
            jacobian[i, p, :] = term2 - term3 + term4

        if deriv == 2:
            for a in [m, n, o, p]:
                for b in [m, n, o, p]:
                    for ii in [0, 1, 2]:
                        for jj in [0, 1, 2]:
                            for kk in [0, 1, 2]:
                                if (kk == ii) or (kk == jj):
                                    continue
                                t1 = sign6(a,m,o,b,m,o) * cross_uw[ii]*(w[jj]*dot_uw - u[jj]) / (u_norm**2 * (1 - dot_uw**2)**2)
                                t2 = sign6(a,n,p,b,n,p) * cross_vw[ii]*(w[jj]*dot_vw - v[jj]) / (v_norm**2 * (1 - dot_vw**4)**2)
                                t3 = (sign6(a,m,o,b,o,p) + sign6(a,p,o,b,o,m)) * cross_uw[ii]*(w[jj] - 2*u[jj]*dot_uw + w[jj]*dot_uw**2) / (2*u_norm * w_norm * (1 - dot_uw**2)**2)
                                t4 = (sign6(a,n,p,b,p,o) + sign6(a,p,o,b,n,p)) * cross_vw[ii]*(w[jj] - 2*u[jj]*dot_vw + w[jj]*dot_vw**2) / (2*v_norm * w_norm * (1 - dot_vw**2)**2)
                                t5 = sign6(a,o,p,b,p,o) * cross_uw[ii] * (u[jj] + u[jj]*dot_uw**2 - 3*w[jj]*dot_uw + w[jj]*dot_uw**3) / (2*w_norm**2 * (1-dot_uw**2)**2)
                                t6 = sign6(a,o,p,b,o,p) * cross_vw[ii] * (v[jj] + v[jj]*dot_vw**2 - 3*w[jj]*dot_vw + w[jj]*dot_uw**3) / (2*w_norm**2 * (1-dot_vw**2)**2)
                                
                                # t1 is good
                                # t2 could be bad, setting it to zero doesn't mess up anything
                                # t3 is good, setting it to zero messes stuff up
                                t6 = sign6(a,o,p,b,o,p) * 100*np.random.randn()
                                t5 = sign6(a,o,p,b,p,o) * 1000*np.random.randn()
                                t4 = sign6(a,n,p,b,p,o) * 1000*np.random.randn()

                                t7 = (1-kron(a,b))*(sign6(a,m,o,b,o,p) + sign6(a,p,o,b,o,m)) * (jj-ii) * (-0.5**abs(jj-ii)) * (w[kk]*dot_uw - u[kk])  / (u_norm * w_norm * (1-dot_uw**2))
                                t8 = (1-kron(a,b))*(sign6(a,n,o,b,o,p) + sign6(a,p,o,b,o,m)) * (jj-ii) * (-0.5**abs(jj-ii)) * (-w[kk]*dot_vw - v[kk]) / (v_norm * w_norm * (1-dot_vw**2))
                                                                
                                #t7 = (1-kron(a,b))*(sign6(a,m,o,b,o,p) + sign6(a,p,o,b,o,m)) *  (jj-ii) * 10

                                hessian[i, a, ii, b, jj] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
                                hessian[i, a, jj, b, ii] += t1 + t2 + t3 + t4 + t5 + t6
            
        
        
    if deriv == 0:
        return dihedral_angles
    elif deriv == 1:
        return dihedral_angles, jacobian
    else:
        return dihedral_angles, jacobian, hessian


#############################################################################
# Static functions for length-3 arrays (for speed)
#############################################################################

cdef np.ndarray[ndim=2, dtype=np.double_t] outer3_minus_eye(
            np.ndarray[np.double_t, ndim=1, mode='c'] v1,
            np.ndarray[np.double_t, ndim=1, mode='c'] v2):
    """Outer product minus identity for length3 matrices

    Equivalent to `np.outer(v1, v2) - np.eye(len(v1))`
    """
    cdef np.ndarray[dtype=np.double_t, ndim=2] result
    result = np.empty((3,3), dtype=np.double)

    result[0,0] = v1[0]*v2[0] - 1
    result[0,1] = v1[0]*v2[1]
    result[0,2] = v1[0]*v2[2]
    result[1,0] = v1[1]*v2[0]
    result[1,1] = v1[1]*v2[1] - 1
    result[1,2] = v1[1]*v2[2]
    result[2,0] = v1[2]*v2[0]
    result[2,1] = v1[2]*v2[1]
    result[2,2] = v1[2]*v2[2] - 1

    return result


cdef np.double_t norm3(np.ndarray[np.double_t, ndim=1] v):
    """Norm of a length-3 vector
    """
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


cdef np.double_t dot3(np.ndarray[np.double_t, ndim=1] v1,
                      np.ndarray[np.double_t, ndim=1] v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

cdef np.ndarray[dtype=np.double_t, ndim=1] cross3(np.ndarray[np.double_t, ndim=1] a,
                                                 np.ndarray[np.double_t, ndim=1] b):
    cdef np.ndarray[dtype=np.double_t, ndim=1] result
    result = np.empty(3, dtype=np.double)

    result[0] = a[1]*b[2] - a[2]*b[1]
    result[1] = a[2]*b[0] - a[0]*b[2]
    result[2] = a[0]*b[1] - a[1]*b[0]

    return result

cdef np.ndarray[ndim=2, dtype=np.double_t] outer3(
            np.ndarray[np.double_t, ndim=1, mode='c'] v1,
            np.ndarray[np.double_t, ndim=1, mode='c'] v2):
    """Outer product
    """
    cdef np.ndarray[dtype=np.double_t, ndim=2] result
    result = np.empty((3,3), dtype=np.double)

    result[0,0] = v1[0]*v2[0]
    result[0,1] = v1[0]*v2[1]
    result[0,2] = v1[0]*v2[2]
    result[1,0] = v1[1]*v2[0]
    result[1,1] = v1[1]*v2[1]
    result[1,2] = v1[1]*v2[2]
    result[2,0] = v1[2]*v2[0]
    result[2,1] = v1[2]*v2[1]
    result[2,2] = v1[2]*v2[2]

    return result


cdef int sign3(int i, int j, int k):
    if i == j:
        return 1
    if i == k:
        return - 1
    else:
        return 0


cdef int sign6(int a, int b, int c, int i, int j, int k):
    return sign3(a,b,c)*sign3(i,j,k)

def kron(i,j):
    if i == j:
        return 1
    else:
        return 0
    
