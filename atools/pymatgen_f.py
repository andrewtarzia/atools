#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for pymatgen usage.

Author: Andrew Tarzia

Date Created: 30 Jan 2020
"""

def get_element_sites(molecule, atomic_no):
    """
    Get the index of sites in the molecule with the desired element.

    Parameters
    ----------
    molecule : :class:`pymatgen.Molecule`
        Molecule to get sites of.

    atomic_no : :class:`int`
        Atomic number of desired element.

    Returns
    -------
    idxs : :class:`list` of :class:`int`
        List of site indices.

    """

    idxs = []

    for i, atom in enumerate(molecule.sites):
        Z = atom.specie.Z
        if Z == atomic_no:
            idxs.append(i)

def calculate_site_order_values(structure, site):
    return idxs
    """
    Calculate order parameters around metal centres.

    Parameters
    ----------

    Returns
    -------

    return results
