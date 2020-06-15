#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions for calculating measures of ligand distortion.

Author: Andrew Tarzia

Date Created: 29 May 2020

"""

import json
import stk
from os.path import exists

from .IO_tools import read_gfnx2xtb_eyfile
from .stko_f import calculate_energy
from .ligand_calculations import (
    calculate_bite_angle,
    calculate_NN_distance,
    get_furthest_pair_FGs,
)
from .rdkit_f import get_query_atom_ids
from .calculations import get_dihedral


def calculate_ligand_SE(
    org_ligs,
    smiles_keys,
    output_json,
    file_prefix=None
):
    """
    Calculate the strain energy of each ligand in the cage.

    Parameters
    ----------
    org_lig : :class:`dict` of :class:`stk.BuildingBlock`
        Dictionary of building blocks where the key is the file name,
        and the value is the stk building block.

    smiles_keys : :class:`dict` of :class:`int`
        Key is the linker smiles, value is the idx of that smiles.

    output_json : :class:`str`
        File name to save output to to avoid reruns.

    file_prefix : :class:`str`, optional
        Prefix to file name of each output ligand structure.
        Eventual file name is:
        "file_prefix"{number of atoms}_{idx}_{i}.mol
        Where `idx` determines if a molecule is unique by smiles.

    Returns
    -------
    strain_energies : :class:`dict`
        Strain energies for each ligand.

    """

    # Check if output file exists.
    if not exists(output_json):
        strain_energies = {}
        # Iterate over ligands.
        for lig in org_ligs:
            stk_lig = org_ligs[lig]
            ey_file = lig.replace('mol', 'ey')
            smiles_key = stk.Smiles().get_key(stk_lig)
            idx = smiles_keys[smiles_key]
            sgt = str(stk_lig.get_num_atoms())
            # Get optimized ligand name that excludes any cage
            # information.
            if file_prefix is None:
                filename_ = f'organic_linker_s{sgt}_{idx}_opt.mol'
                opt_lig_ey = f'organic_linker_s{sgt}_{idx}_opt.ey'
                opt_lig_n = f'organic_linker_s{sgt}_{idx}_opt'
            else:
                filename_ = f'{file_prefix}{sgt}_{idx}_opt.mol'
                opt_lig_ey = f'{file_prefix}{sgt}_{idx}_opt.ey'
                opt_lig_n = f'{file_prefix}{sgt}_{idx}_opt'

            # Calculate energy of extracted ligand.
            if not exists(ey_file):
                calculate_energy(
                    name=lig.replace('.mol', ''),
                    mol=stk_lig,
                    ey_file=ey_file
                )
            # Read energy.
            # kJ/mol.
            E_extracted = read_gfnx2xtb_eyfile(ey_file)

            # Calculate energy of optimised ligand.
            # Load in lowest energy conformer.
            opt_mol = stk.BuildingBlock.init_from_file(
                filename_
            )
            if not exists(opt_lig_ey):
                calculate_energy(
                    name=opt_lig_n,
                    mol=opt_mol,
                    ey_file=opt_lig_ey
                )
            # Read energy.
            # kJ/mol.
            E_free = read_gfnx2xtb_eyfile(opt_lig_ey)
            # Add to list the strain energy:
            # (E(extracted) - E(optimised/free))
            lse = E_extracted - E_free
            # kJ/mol.
            strain_energies[lig] = lse

        # Write data.
        with open(output_json, 'w') as f:
            json.dump(strain_energies, f)

    # Get data.
    with open(output_json, 'r') as f:
        strain_energies = json.load(f)

    return strain_energies


def calculate_abs_imine_torsions(org_ligs):
    """
    Calculate the imine torsion of all ligands in the cage.

    """

    torsions = {}
    # Iterate over ligands.
    for lig in org_ligs:
        stk_lig = org_ligs[lig]
        print(lig)
        # Find torsions.
        smarts = '[#6]-[#7X2]=[#6X3H1]-[#6X3]'
        rdkit_mol = stk_lig.to_rdkit_mol()
        query_ids = get_query_atom_ids(smarts, rdkit_mol)
        # Calculate torsional angle for all imines.
        torsion_list = []
        for atom_ids in query_ids:
            torsion = get_dihedral(
                pt1=tuple(
                    stk_lig.get_atomic_positions(atom_ids[0])
                )[0],
                pt2=tuple(
                    stk_lig.get_atomic_positions(atom_ids[1])
                )[0],
                pt3=tuple(
                    stk_lig.get_atomic_positions(atom_ids[2])
                )[0],
                pt4=tuple(
                    stk_lig.get_atomic_positions(atom_ids[3])
                )[0]
            )
            torsion_list.append(abs(torsion))

        # Degrees
        torsions[lig] = torsion_list

    return torsions


def calculate_ligand_planarities(org_ligs):
    """
    Calculate the change in planarity of the core of all ligands.

    """

    # Iterate over each ligand and find core based on the input
    # molecule and its FGs (i.e. the core is the parts of the
    # input molecule that is not part of the FGs).

    # Calculate planarity of the cores compared to input molecule.

    # Save to list.

    return []


def calculate_deltaangle_distance(
    org_ligs,
    smiles_keys,
    fg_factory,
    file_prefix=None
):
    """
    Calculate the change of bite angle of each ligand in the cage.

    This function will not work for cages built from FGs other than
    metals + NPyridine and metals + NTriazole.

    Parameters
    ----------
    org_lig : :class:`dict` of :class:`stk.BuildingBlock`
        Dictionary of building blocks where the key is the file name,
        and the value is the stk building block.

    smiles_keys : :class:`dict` of :class:`int`
        Key is the linker smiles, value is the idx of that smiles.

    fg_factory :
    :class:`iterable` of :class:`stk.FunctionalGroupFactory`
        Functional groups to asign to molecules.
        NN_distance calculator will not work for cages built from FGs
        other than metals + NPyridine and metals + NTriazole.

    file_prefix : :class:`str`, optional
        Prefix to file name of each output ligand structure.
        Eventual file name is:
        "file_prefix"{number of atoms}_{idx}_{i}.mol
        Where `idx` determines if a molecule is unique by smiles.

    Returns
    -------
    delta_angles : :class:`dict`
        Bite angle in cage - free optimised ligand for each ligand.
        Output is absolute values.

    """

    delta_angles = {}
    # Iterate over ligands.
    for lig in org_ligs:
        stk_lig = org_ligs[lig]
        smiles_key = stk.Smiles().get_key(stk_lig)
        idx = smiles_keys[smiles_key]
        sgt = str(stk_lig.get_num_atoms())
        # Get optimized ligand name that excludes any cage
        # information.
        if file_prefix is None:
            filename_ = f'organic_linker_s{sgt}_{idx}_opt.mol'
        else:
            filename_ = f'{file_prefix}{sgt}_{idx}_opt.mol'

        _in_cage = stk.BuildingBlock.init_from_molecule(
            stk_lig,
            functional_groups=fg_factory
        )
        _in_cage = _in_cage.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(_in_cage)
        )

        _free = stk.BuildingBlock.init_from_file(
            filename_,
            functional_groups=fg_factory
        )
        _free = _free.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(_free)
        )
        angle_in_cage = calculate_bite_angle(bb=_in_cage)
        angle_free = calculate_bite_angle(bb=_free)

        delta_angles[lig] = abs(angle_in_cage - angle_free)

    return delta_angles


def calculate_deltann_distance(
    org_ligs,
    smiles_keys,
    fg_factory,
    file_prefix=None
):
    """
    Calculate the change of NN distance of each ligand in the cage.

    This function will not work for cages built from FGs other than
    metals + NPyridine and metals + NTriazole.

    Parameters
    ----------
    org_lig : :class:`dict` of :class:`stk.BuildingBlock`
        Dictionary of building blocks where the key is the file name,
        and the value is the stk building block.

    smiles_keys : :class:`dict` of :class:`int`
        Key is the linker smiles, value is the idx of that smiles.

    fg_factory :
    :class:`iterable` of :class:`stk.FunctionalGroupFactory`
        Functional groups to asign to molecules.
        NN_distance calculator will not work for cages built from FGs
        other than metals + NPyridine and metals + NTriazole.

    file_prefix : :class:`str`, optional
        Prefix to file name of each output ligand structure.
        Eventual file name is:
        "file_prefix"{number of atoms}_{idx}_{i}.mol
        Where `idx` determines if a molecule is unique by smiles.

    Returns
    -------
    delta_nns : :class:`dict`
        NN distance in cage - free optimised ligand for each ligand.
        Output is absolute values.

    """

    delta_nns = {}
    # Iterate over ligands.
    for lig in org_ligs:
        stk_lig = org_ligs[lig]
        smiles_key = stk.Smiles().get_key(stk_lig)
        idx = smiles_keys[smiles_key]
        sgt = str(stk_lig.get_num_atoms())
        # Get optimized ligand name that excludes any cage
        # information.
        if file_prefix is None:
            filename_ = f'organic_linker_s{sgt}_{idx}_opt.mol'
        else:
            filename_ = f'{file_prefix}{sgt}_{idx}_opt.mol'

        _in_cage = stk.BuildingBlock.init_from_molecule(
            stk_lig,
            functional_groups=fg_factory
        )
        _in_cage = _in_cage.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(_in_cage)
        )

        _free = stk.BuildingBlock.init_from_file(
            filename_,
            functional_groups=fg_factory
        )
        _free = _free.with_functional_groups(
            functional_groups=get_furthest_pair_FGs(_free)
        )

        nn_in_cage = calculate_NN_distance(bb=_in_cage)
        nn_free = calculate_NN_distance(bb=_free)

        delta_nns[lig] = abs(nn_in_cage - nn_free)

    return delta_nns
