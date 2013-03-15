import numpy as np

def rmsd(query, target, operator=True):
    """Compute the RMSD
    
    Parameters
    ----------
    return_matricies : bool
        return the rotation and translation matricies too.
    """
    if not query.ndim == 2:
        raise ValueError('query must be 2d')
    if not target.ndim == 2:
        raise ValueError('target must be 2d')
    n_atoms, three = query.shape
    if not three == 3:
        raise ValueError('query second dimension must be 3')
    n_atoms, three = target.shape
    if not three == 3:
        raise ValueError('target second dimension must be 3')
    if not query.shape[0] == target.shape[0]:
        raise ValueError('query and target must have same number of atoms')

    # centroids
    m_query = np.mean(query, axis=0)
    m_target = np.mean(target, axis=0)

    # centered
    c_query = query - m_query
    c_target = target - m_target

    error_0 = np.sum(c_query**2) + np.sum(c_target**2)
    
    A = np.dot(c_query.T, c_target)
    u, s, v = np.linalg.svd(A)
    
    d = np.diag([1, 1, np.sign(np.linalg.det(A))])
        
    rmsd = np.sqrt(np.abs(error_0 - (2.0 * np.sum(s))) / n_atoms)
    
    if operator:
        rotation_matrix = np.dot(v.T, u.T).T
        translation_matrix = m_query - np.dot(m_target, rotation_matrix)
        return rmsd, AlignOperator(rotation_matrix, translation_matrix)
    
    return rmsd


class AlignOperator(object):
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans
    
    def __call__(self, matrix):
        return np.dot(matrix, self.rot) + self.trans

if __name__ == '__main__':
    N = 40
    query = np.arange(N)[:, np.newaxis] * np.random.randn(N,3)
    target = np.arange(N)[:, np.newaxis] * np.random.randn(N,3)

    dist, op =  rmsd(query, target)
    print 'my rmsd        ', dist
    
    from msmbuilder.metrics import RMSD
    _rmsdcalc = RMSD()
    t0 = RMSD.TheoData(query[np.newaxis, :, :])
    t1 = RMSD.TheoData(target[np.newaxis, :, :])
    print 'msmbuilder rmsd', _rmsdcalc.one_to_all(t0, t1, 0)[0]
    
    print np.sqrt(np.sum(np.square(target - op(query))) / N)
    
    

def progressive_align(xyzlist):
    """Align each frame to the one behind it.
    
    Uses a simple version of the Kabsch algorithm, as described on
    wikipedia: https://en.wikipedia.org/wiki/Kabsch_algorithm
    
    Parameters
    ----------
    xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The input cartesian coordinates.
    
    Returns
    -------
    c_xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3]
        The outut cartesian coordinates, after alignment.
    """

    centroids = np.mean(xyzlist, axis=1)
    
    # for i in range(len(xyzlist)):
    #     xyzlist[i] -= centroids[i]
    
    # center the conformations
    c_xyzlist = xyzlist - centroids[:, np.newaxis, :]
    
    for i in range(1, len(xyzlist)):
        A = np.dot(c_xyzlist[i].T, c_xyzlist[i-1])
        v, s, wt = np.linalg.svd(A)



        rotation_matrix = np.dot(v, wt)

        c_xyzlist[i] = np.dot(c_xyzlist[i], rotation_matrix)
        print np.linalg.norm(c_xyzlist[i] - c_xyzlist[i-1])
    
    return xyzlist

# def main():
#     from io import XYZFile
#     xyzlist, _ = XYZFile('../reaction_015.xyz').read_trajectory()
#     progressive_align(xyzlist)
#     XYZFile('../reaction_015.align.xyz', 'w').write_trajectory(xyzlist, _)
#     
# if __name__ == '__main__':
#     main()
