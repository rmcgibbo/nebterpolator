import numpy as np

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
    
    # center the conformations
    c_xyzlist = xyzlist - centroids[:, np.newaxis, :]
    
    for i in range(1, len(xyzlist)):
        A = np.dot(c_xyzlist[i].T, c_xyzlist[i-1])  # covariance
        v, s, w = np.linalg.svd(A)

        # do we need to flip our coord system?
        d = np.linalg.det(np.dot(w, v.T))
        m = np.eye(3,3)
        m[d,d] = d
        
        rotation_matrix = np.dot(np.dot(w, m), v.T)

        c_xyzlist[i] = np.dot(c_xyzlist[i], rotation_matrix.T)
    
    return c_xyzlist
