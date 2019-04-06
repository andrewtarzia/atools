#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for reading/writing structure files

Author: Andrew Tarzia

Date Created: 15 Mar 2019
"""

from ase.io import read
from ase.io.xyz import write_xyz
from pymatgen.io.cif import CifParser


def convert_CIF_2_PDB(file):
    '''Convert CIF to PDB file, save and return structure.

    '''
    pdb_file = file.replace('.cif', '.pdb')
    print('converting:', file, 'to', pdb_file)
    structure = read(file)
    # view(structure)
    # input()
    structure.write(pdb_file)
    print('conversion done.')
    return pdb_file, structure


def convert_PDB_2_XYZ(file, comment=None):
    '''Convert PDB to standard (NOT exteneded) XYZ file, save and
    return structure

    '''
    xyz_file = file.replace('.pdb', '.xyz')
    print('converting:', file, 'to', xyz_file)
    structure = read(file)
    # view(structure)
    # input()
    if comment is None:
        cmt = 'This is an XYZ structure.'
    else:
        cmt = comment
    write_xyz(xyz_file, images=structure, comment=cmt,
              columns=['symbols', 'positions'])
    print('conversion done.')
    return xyz_file, structure


def read_cif_pmg(file, primitive=False):
    '''A function to read CIFs with pymatgen and suppress warnings.

    '''
    s = CifParser(file, occupancy_tolerance=100)
    struct = s.get_structures(primitive=primitive)[0]
    return struct