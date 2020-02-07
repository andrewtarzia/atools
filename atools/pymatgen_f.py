#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for pymatgen usage.

Author: Andrew Tarzia

Date Created: 30 Jan 2020

"""

from pymatgen.analysis.local_env import (
    LocalStructOrderParams,
)
from pymatgen.core import Specie


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

    return idxs


def calculate_sites_order_values(
    molecule,
    site_idxs,
    target_species_type=None,
    neigh_idxs=None
):
    """
    Calculate order parameters around metal centres.

    Parameters
    ----------
    molecule : :class:`pmg.Molecule` or :class:`pmg.Structure`
        Pymatgen (pmg) molecule/structure to analyse.

    site_idxs : :class:`list` of :class:`int`
        Atom ids of sites to calculate OP of.

    target_species_type : :class:`str`
        Target neighbour element to use in OP calculation.
        Defaults to :class:`NoneType` if no target species is known.

    neigh_idxs : :class:`list` of :class:`list` of :class:`int`
        Neighbours of each atom in site_idx. Ordering is important.
        Defaults to :class:`NoneType` for when using
        :class:`pmg.Structure` - i.e. a structure with a lattice.

    Returns
    -------
    results : :class:`dict`
        Dictionary of format
        site_idx: dict of order parameters
        {
            `oct`: :class:`float`,
            `sq_plan`: :class:`float`,
            `q2`: :class:`float`,
            `q4`: :class:`float`,
            `q6`: :class:`float`
        }.

    """

    results = {}

    if target_species_type is None:
        targ_species = None
    else:
        targ_species = Specie(target_species_type)

    # Define local order parameters class based on desired types.
    types = [
        'oct',  # Octahedra OP.
        'sq_plan',  # Square planar envs.
        'q2',  # l=2 Steinhardt OP.
        'q4',  # l=4 Steinhardt OP.
        'q6',  # l=6 Steinhardt OP.
    ]
    loc_ops = LocalStructOrderParams(
        types=types,
    )
    if neigh_idxs is None:
        for site in site_idxs:
            site_results = loc_ops.get_order_parameters(
                structure=molecule,
                n=site,
                target_spec=[targ_species]
            )
            results[site] = {i: j for i, j in zip(types, site_results)}
    else:
        for site, neigh in zip(site_idxs, neigh_idxs):
            site_results = loc_ops.get_order_parameters(
                structure=molecule,
                n=site,
                indices_neighs=neigh,
                target_spec=targ_species
            )
            results[site] = {i: j for i, j in zip(types, site_results)}

    return results
