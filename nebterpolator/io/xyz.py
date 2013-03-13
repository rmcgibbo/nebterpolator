import itertools
import numpy as np


class XYZFormatError(Exception):
    pass


class XYZFile(object):
    def __init__(self, filename):
        """Open a .xyz format file
        """
        self._handle = open(filename)

    def read_frame(self):
        """Read a single molecule/frame from an xyz file

        Returns
        -------
        xyz : np.ndarray, shape=[n_atoms, 3], dtype=float
            The cartesian coordinates
        atom_names : list of strings
            A list of the names of the atoms
        """
        try:
            n_atoms = self._handle.readline()
            if n_atoms == '':
                raise EOFError('The end of the file was reached')
            n_atoms = int(n_atoms)
        except ValueError:
            raise XYZFormatError(('The number of atoms wasn\'t parsed '
                                  'correctly.'))

        comment = self._handle.readline()
        atom_names = [None for i in range(n_atoms)]
        xyz = np.zeros((n_atoms, 3))

        for i in xrange(n_atoms):
            line = self._handle.readline().split()
            if len(line) != 4:
                raise XYZFormatError('line was not 4 elemnts: %s' % str(line))

            atom_names[i] = line[0]
            try:
                xyz[i] = np.array([float(e) for e in line[1:]], dtype=float)
            except:
                raise XYZFormatError(('The coordinates were not correctly '
                                      'parsed.'))

        return xyz, atom_names

    def read_trajectory(self):
        """Read all of the frames from a xyzfile

        Returns
        -------
        xyzlist : np.ndarray, shape=[n_frames, n_atoms, 3], dtype=float
            The cartesian coordinates
        atom_names : list of strings
            A list of the names of the atoms
        """
        xyz, atom_names = self.read_frame()

        np_atom_names = np.array(atom_names)
        xyzlist = [xyz]

        for i in itertools.count(1):
            try:
                xyz, tmp_atom_names = self.read_frame()
                xyzlist.append(xyz)
                if not np.all(np_atom_names == np.array(tmp_atom_names)):
                    raise XYZFormatError(('Frame %d does not contain the same'
                                          'atoms as the other frames' % i))
            except EOFError:
                break

        return np.array(xyzlist), atom_names

    def __del__(self):
        self.close()

    def close(self):
        self._handle.close()


def main():
    f = XYZFile('/Users/rmcgibbo/projects/nebterpolator/reaction_015.xyz')
    print f.read_trajectory()

if __name__ == '__main__':
    main()
