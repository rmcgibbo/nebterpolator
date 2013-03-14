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
    
    # for i in range(len(xyzlist)):
    #     xyzlist[i] -= centroids[i]
    
    # center the conformations
    c_xyzlist = xyzlist - centroids[:, np.newaxis, :]
    
    for i in range(1, len(xyzlist)):
        A = np.dot(c_xyzlist[i].T, c_xyzlist[i-1])
        v, s, wt = np.linalg.svd(A)

        # # do we need to flip our coord system?
        if  np.linalg.det(wt) * np.linalg.det(v) < 0:
            s[-1] *= -1
            v[:, -1] *= -1

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
