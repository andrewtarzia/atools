#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions for calculating measures of metal centre distortion.

Author: Andrew Tarzia

Date Created: 18 Mar 2019
"""

import numpy as np
from itertools import combinations

from .calculations import (
    get_dihedral,
    shortest_distance_to_plane,
    angle_between
)
from .pymatgen_f import calculate_sites_order_values
from .IO_tools import convert_stk_to_pymatgen
from .stk_f import get_atom_distance, get_stk_bond_angle


def get_square_planar_distortion(mol, metal, bonder):
    """
    Calculate measures of distortion of a square planer metal.

    Parameters
    ----------
    mol : :class:`stk.ConstructedMolecule`
        stk molecule to analyse.

    metal : :class:`int`
        Element number of metal atom.

    bonder : :class:`int`
        Element number of atoms bonded to metal.

    Returns
    -------
    results : :class:`dict`
        Dictionary containing 'bond_lengths', 'angles', 'torsions' and
        'plane_dev'.

    """

    results = {
        'bond_lengths': [],
        'angles': [],
        'torsions': [],
        'plane_dev': [],
        'plane_angle_avg': [],
        'plane_angle_std': []
    }

    # Find metal atoms.
    metal_atoms = []
    for atom in mol.get_atoms():
        if atom.get_atomic_number() == metal:
            metal_atoms.append(atom)

    # Find bonders.
    metal_bonds = []
    ids_to_metals = []
    for bond in mol.get_bonds():
        if bond.get_atom1() in metal_atoms:
            metal_bonds.append(bond)
            ids_to_metals.append(bond.get_atom2().get_id())
        elif bond.get_atom2() in metal_atoms:
            metal_bonds.append(bond)
            ids_to_metals.append(bond.get_atom1().get_id())

    # Calculate bond lengths.
    for bond in metal_bonds:
        results['bond_lengths'].append(
            get_atom_distance(
                molecule=mol,
                atom1_id=bond.get_atom1().get_id(),
                atom2_id=bond.get_atom2().get_id(),
            )
        )

    # Calculate bond angles.
    for bonds in combinations(metal_bonds, r=2):
        bond1, bond2 = bonds
        bond1_atoms = [bond1.get_atom1(), bond1.get_atom2()]
        bond2_atoms = [bond2.get_atom1(), bond2.get_atom2()]
        pres_atoms = list(set(bond1_atoms + bond2_atoms))
        # If there are more than 3 atoms, implies two
        # independant bonds.
        if len(pres_atoms) > 3:
            continue
        for atom in pres_atoms:
            if atom in bond1_atoms and atom in bond2_atoms:
                idx2 = atom.get_id()
            elif atom in bond1_atoms:
                idx1 = atom.get_id()
            elif atom in bond2_atoms:
                idx3 = atom.get_id()

        angle = np.degrees(get_stk_bond_angle(
            mol=mol,
            atom1_id=idx1,
            atom2_id=idx2,
            atom3_id=idx3,
        ))
        if angle < 120:
            results['angles'].append(angle)

    # Calculate torsion.
    for metal_atom in metal_atoms:
        torsion_ids = []
        for bond in metal_bonds:
            if metal_atom.get_id() == bond.get_atom1().get_id():
                torsion_ids.append(bond.get_atom2().get_id())
            elif metal_atom.get_id() == bond.get_atom2().get_id():
                torsion_ids.append(bond.get_atom1().get_id())
        atom_positions = [
            i for i in mol.get_atomic_positions(atom_ids=torsion_ids)
        ]
        results['torsions'].append(abs(get_dihedral(*atom_positions)))

    # Calculate deviation of metal from bonder plane.
    for metal_atom in metal_atoms:
        binder_atom_ids = [metal_atom.get_id()]
        for bond in metal_bonds:
            if metal_atom.get_id() == bond.get_atom1().get_id():
                binder_atom_ids.append(bond.get_atom2().get_id())
            elif metal_atom.get_id() == bond.get_atom2().get_id():
                binder_atom_ids.append(bond.get_atom1().get_id())
        centroid = mol.get_centroid(atom_ids=binder_atom_ids)
        normal = mol.get_plane_normal(atom_ids=binder_atom_ids)
        # Plane of equation ax + by + cz = d.
        binder_atom_plane = np.append(normal, np.sum(normal*centroid))
        # Define the plane deviation as the sum of the distance of all
        # atoms from the plane defined by all atoms.
        plane_dev = sum([
            shortest_distance_to_plane(
                binder_atom_plane,
                tuple(mol.get_atomic_positions(atom_ids=i), )[0]
            )
            for i in binder_atom_ids
        ])
        results['plane_dev'].append(plane_dev)

    # Calculate N-Pd bond vector and CNC plane angle.
    for metal_atom in metal_atoms:
        plane_angles = []
        # Get metal position.
        metal_position = mol.get_centroid(
            atom_ids=[metal_atom.get_id()]
        )
        # Iterate over metal bonds.
        for bond in metal_bonds:
            if metal_atom.get_id() == bond.get_atom1().get_id():
                N_atom_id = bond.get_atom2().get_id()
            elif metal_atom.get_id() == bond.get_atom2().get_id():
                N_atom_id = bond.get_atom1().get_id()
            else:
                continue
            # Get MN vector.
            N_position = mol.get_centroid(atom_ids=[N_atom_id])
            MN_vector = N_position - metal_position
            # Get CNC atom ids.
            CNC_atom_ids = [N_atom_id]
            for bond in mol.get_bonds():
                if metal_atom.get_id() in [
                    bond.get_atom1().get_id(),
                    bond.get_atom2().get_id()
                ]:
                    continue
                if bond.get_atom1().get_id() == N_atom_id:
                    CNC_atom_ids.append(bond.get_atom2().get_id())
                elif bond.get_atom2().get_id() == N_atom_id:
                    CNC_atom_ids.append(bond.get_atom1().get_id())

            # Get CNC plane.
            centroid = mol.get_centroid(atom_ids=CNC_atom_ids)
            CNC_plane_normal = mol.get_plane_normal(
                atom_ids=CNC_atom_ids
            )
            # Calculate angle between CNC plane and MN vector.
            pa = np.degrees(angle_between(MN_vector, CNC_plane_normal))
            plane_angles.append(pa)

        # Define the plane angle of a metal centre as the sum of all
        # plane angles of 4 coordinated atoms.
        plane_angle_avg = np.average([i for i in plane_angles])
        plane_angle_std = np.std([i for i in plane_angles])
        results['plane_angle_avg'].append(plane_angle_avg)
        results['plane_angle_std'].append(plane_angle_std)

    return results


def get_order_values(mol, metal, per_site=False):
    """
    Calculate order parameters around metal centres.

    Parameters
    ----------
    mol : :class:`stk.ConstructedMolecule`
        stk molecule to analyse.

    metal : :class:`int`
        Element number of metal atom.

    per_site : :class:`bool`
        Defaults to False. True if the OPs for each site are desired.

    Returns
    -------
    results : :class:`dict`
        Dictionary of order parameter max/mins/averages if `per_site`
        is False.

    """

    pmg_mol = convert_stk_to_pymatgen(stk_mol=mol)
    # Get sites of interest and their neighbours.
    sites = []
    neighs = []
    for atom in mol.get_atoms():
        if atom.get_atomic_number() == metal:
            sites.append(atom.get_id())
            bonds = [
                i
                for i in mol.get_bonds()
                if i.get_atom1().get_id() == atom.get_id()
                or i.get_atom2().get_id() == atom.get_id()
            ]
            a_neigh = []
            for b in bonds:
                if b.get_atom1().get_id() == atom.get_id():
                    a_neigh.append(b.get_atom2().get_id())
                elif b.get_atom2().get_id() == atom.get_id():
                    a_neigh.append(b.get_atom1().get_id())
            neighs.append(a_neigh)

    order_values = calculate_sites_order_values(
        molecule=pmg_mol,
        site_idxs=sites,
        neigh_idxs=neighs
    )
    if per_site:
        results = order_values
        return results
    else:
        # Get max, mins and averages of all OPs for the whole molecule.
        OPs = [order_values[i].keys() for i in order_values][0]
        OP_lists = {}
        for OP in OPs:
            OP_lists[OP] = [order_values[i][OP] for i in order_values]

        results = {
            # OP: (min, max, avg)
            i: {
                'min': min(OP_lists[i]),
                'max': max(OP_lists[i]),
                'avg': np.average(OP_lists[i])
            }
            for i in OP_lists
        }

        return results
