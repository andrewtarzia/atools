#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for ASE usage.

Author: Andrew Tarzia

Date Created: 12 Jun 2020

"""

from ase.io import read


def load_periodic_pdb(filename):
    """
    Load pdb file to periodic ASE object.

    """

    ase_atoms = read(filename)
    ase_atoms.set_pbc(True)

    return ase_atoms
