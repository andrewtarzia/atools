#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for stk usage.

Author: Andrew Tarzia

Date Created: 18 Mar 2019
"""

from glob import glob
from os.path import exists
import stk
import numpy as np
from itertools import combinations

from .calculations import (
    get_dihedral,
    shortest_distance_to_plane,
    angle_between
)


def build_ABCBA(core, liga, link, flippedlink=False):
    """
    Build ABCBA ligand using linear stk polymer.

    Polymer structure:
        ligand -- linker -- core -- linker -- ligand

    Parameters
    ----------
    core : :class:`stk.BuildingBlock`
        The molecule to use as core.
    liga : :class:`stk.BuildingBlock`
        The molecule to use as ligand.
    link : :class:`stk.BuildingBlock`
        The molecule to use as linker.
    flippedlink : :class:`bool`
        `True` to flip the linker molecules. Defaults to `False`.

    Returns
    -------
    polymer : :class:`stk.ConstructedMolecule`
        Built molecule pre optimisation.

    """
    if flippedlink is False:
        orientation = (0, 0, 0, 1, 1)
    elif flippedlink is True:
        orientation = (0, 1, 0, 0, 1)
    polymer = stk.ConstructedMolecule(
        building_blocks=[liga, link, core],
        topology_graph=stk.polymer.Linear(
            repeating_unit='ABCBA',
            num_repeating_units=1,
            orientations=orientation,
            num_processes=1
        )
    )
    return polymer


def build_ABA(core, liga):
    """
    Build ABA ligand using linear stk.Polymer().

    Polymer structure:
        ligand -- core -- ligand

    Parameters
    ----------
    core : :class:`stk.BuildingBlock`
        The molecule to use as core.
    liga : :class:`stk.BuildingBlock`
        The molecule to use as ligand.

    Returns
    -------
    polymer : :class:`stk.ConstructedMolecule`
        Built molecule pre optimisation.

    """
    polymer = stk.ConstructedMolecule(
        building_blocks=[liga, core],
        topology_graph=stk.polymer.Linear(
            repeating_unit='ABA',
            num_repeating_units=1,
            orientations=(0, 0, 1),
            num_processes=1
        )
    )
    return polymer


def build_population(directory, fgs=None, suffix='.mol'):
    """
    Build population of BuilingBlocks from directory.

    Parameters
    ----------
    directory : :class:`str`
        Directory containing molecule files.
    fgs : :class:`list` of :class:`str`
        Functional groups of BuildingBlocks. Defaults to bromine.
    suffix : :class:`str`
        File suffix to use.

    Returns
    -------
    popn : :class:`stk.Population`
        Population of BuildingBlocks.

    """
    if fgs is None:
        fgs = ['bromine']

    mols = []
    for file in sorted(glob(directory + '*' + suffix)):
        mol = stk.BuildingBlock.init_from_file(
            path=file,
            functional_groups=fgs,
            use_cache=False
        )
        mol.name = file.rstrip(suffix).replace(directory, '')
        mols.append(mol)
    popn = stk.Population(*mols)
    return popn


def topo_2_property(topology, property):
    """
    Returns properties of a topology for a given topology name.

    Properties:
        'stk_func' - gives the stk topology function for building cages
        'stoich' - gives the stoichiometries of both building blocks
            assuming that the first building block has the larger
            number of functional groups.
        'noimines' - gives the number of imines formed to build that
            topology
        'expected_wind' - gives the number of windows expected

    Currently defined topologies:
        TwoPlusThree topologies
        ThreePlusThree topologies

    """
    properties = ['stk_func', 'stoich', 'noimines', 'expected_wind']
    if property not in properties:
        raise ValueError(
            f'{property} not defined'
            f'possible properties: {properties}'
            'exitting.'
        )

    dict = {
        '2p3': {
            'stk_func': stk.cage.TwoPlusThree(),
            'stoich': (2, 3),
            'noimines': 6,
            'expected_wind': 3,
        },
        '4p6': {
            'stk_func': stk.cage.FourPlusSix(),
            'stoich': (4, 6),
            'noimines': 12,
            'expected_wind': 4,
        },
        '4p62': {
            'stk_func': stk.cage.FourPlusSix2(),
            'stoich': (4, 6),
            'noimines': 12,
            'expected_wind': 4,
        },
        '6p9': {
            'stk_func': stk.cage.SixPlusNine(),
            'stoich': (6, 9),
            'noimines': 18,
            'expected_wind': 5,
        },
        '8p12': {
            'stk_func': stk.cage.EightPlusTwelve(),
            'stoich': (8, 12),
            'noimines': 24,
            'expected_wind': 6,
        },
        'dodec': {
            'stk_func': stk.cage.TwentyPlusThirty(),
            'stoich': (20, 30),
            'noimines': 60,
            'expected_wind': 12,
        },
        '1p1': {
            'stk_func': stk.cage.OnePlusOne(),
            # BB placements used in amarsh project.
            'bb_positions': {0: [0], 1: [1]},
            'stoich': (1, 1),
            'noimines': 3,
            'expected_wind': 3,
        },
        '4p4': {
            'stk_func': stk.cage.FourPlusFour(),
            # BB placements used in amarsh project.
            'bb_positions': {0: [0, 3, 5, 6], 1: [1, 2, 4, 7]},
            'stoich': (4, 4),
            'noimines': 12,
            'expected_wind': 6,
        },
    }
    if topology not in dict:
        raise ValueError(
            f'properties not defined for {topology}'
            'exitting'
        )
    return dict[topology][property]


def is_porous(pore_diameter, max_window_diameter):
    """
    Returns True if a cage is deemed to be porous.

    A porous cage is defined as having:
        (Computationally-inspired discovery of an unsymmetrical
        porous organic cage)
        1 - p_diam_opt > 3.4
        2 - max window_diameter > 2.8

    """
    if max_window_diameter > 2.8 and pore_diameter > 3.4:
        return True
    else:
        return False


def is_collapsed(topo, pore_diameter, no_window):
    """
    Returns True if a cage is deemed to be collapsed.

    A collapsed cage is defined as having:
        - pore_diam_opt < 2.8 Angstrom (H2 kinetic diameter)
        - number of windows != expected number based on topology.

    """
    expected_wind = topo_2_property(topo, property='expected_wind')
    if expected_wind != no_window:
        return True
    elif pore_diameter < 2.8:
        return True
    else:
        return False


def update_from_rdkit_conf(struct, mol, conf_id):
    """
    Update the structure to match `conf_id` of `mol`.

    Parameters
    ----------
    struct : :class:`stk.Molecule`
        The molecule whoce coordinates are to be updated.
    mol : :class:`rdkit.Mol`
        The :mod:`rdkit` molecule to use for the structure update.
    conf_id : :class:`int`
        The conformer ID of the `mol` to update from.

    Returns
    -------
    :class:`.Molecule`
        The molecule.

    """

    pos_mat = mol.GetConformer(id=conf_id).GetPositions()
    struct.set_position_matrix(pos_mat)
    return struct


def get_stk_bond_angle(mol, atom1_id, atom2_id, atom3_id):
    atom1_pos = np.asarray([
        i for i in mol.get_atom_positions(atom_ids=[atom1_id])
    ][0])
    atom2_pos = np.asarray([
        i for i in mol.get_atom_positions(atom_ids=[atom2_id])
    ][0])
    atom3_pos = np.asarray([
        i for i in mol.get_atom_positions(atom_ids=[atom3_id])
    ][0])
    v1 = atom1_pos - atom2_pos
    v2 = atom3_pos - atom2_pos
    return stk.vector_angle(v1, v2)


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
    for atom in mol.atoms:
        if atom.atomic_number == metal:
            metal_atoms.append(atom)

    # Find bonders.
    metal_bonds = []
    ids_to_metals = []
    for bond in mol.bonds:
        if bond.atom1 in metal_atoms:
            metal_bonds.append(bond)
            ids_to_metals.append(bond.atom2.id)
        elif bond.atom2 in metal_atoms:
            metal_bonds.append(bond)
            ids_to_metals.append(bond.atom1.id)

    # Calculate bond lengths.
    for bond in metal_bonds:
        results['bond_lengths'].append(
            mol.get_atom_distance(
                atom1_id=bond.atom1.id,
                atom2_id=bond.atom2.id,
            )
        )

    # Calculate bond angles.
    for bonds in combinations(metal_bonds, r=2):
        bond1, bond2 = bonds
        bond1_atoms = [bond1.atom1, bond1.atom2]
        bond2_atoms = [bond2.atom1, bond2.atom2]
        pres_atoms = list(set(bond1_atoms + bond2_atoms))
        # If there are more than 3 atoms, implies two
        # independant bonds.
        if len(pres_atoms) > 3:
            continue
        for atom in pres_atoms:
            if atom in bond1_atoms and atom in bond2_atoms:
                idx2 = atom.id
            elif atom in bond1_atoms:
                idx1 = atom.id
            elif atom in bond2_atoms:
                idx3 = atom.id

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
            if metal_atom.id == bond.atom1.id:
                torsion_ids.append(bond.atom2.id)
            elif metal_atom.id == bond.atom2.id:
                torsion_ids.append(bond.atom1.id)
        atom_positions = [
            i for i in mol.get_atom_positions(atom_ids=torsion_ids)
        ]
        results['torsions'].append(abs(get_dihedral(*atom_positions)))

    # Calculate deviation of metal from bonder plane.
    for metal_atom in metal_atoms:
        binder_atom_ids = [metal_atom.id]
        for bond in metal_bonds:
            if metal_atom.id == bond.atom1.id:
                binder_atom_ids.append(bond.atom2.id)
            elif metal_atom.id == bond.atom2.id:
                binder_atom_ids.append(bond.atom1.id)
        centroid = mol.get_centroid(atom_ids=binder_atom_ids)
        normal = mol.get_plane_normal(atom_ids=binder_atom_ids)
        # Plane of equation ax + by + cz = d.
        binder_atom_plane = np.append(normal, np.sum(normal*centroid))
        # Define the plane deviation as the sum of the distance of all
        # atoms from the plane defined by all atoms.
        plane_dev = sum([
            shortest_distance_to_plane(
                binder_atom_plane,
                np.asarray([
                    i for i in mol.get_atom_positions(atom_ids=[i])
                ][0])
            )
            for i in binder_atom_ids
        ])
        results['plane_dev'].append(plane_dev)

    # Calculate N-Pd bond vector and CNC plane angle.
    for metal_atom in metal_atoms:
        plane_angles = []
        # Get metal position.
        metal_position = mol.get_centroid(atom_ids=[metal_atom.id])
        # Iterate over metal bonds.
        for bond in metal_bonds:
            if metal_atom.id == bond.atom1.id:
                N_atom_id = bond.atom2.id
            elif metal_atom.id == bond.atom2.id:
                N_atom_id = bond.atom1.id
            else:
                continue
            # Get MN vector.
            N_position = mol.get_centroid(atom_ids=[N_atom_id])
            print(metal_position, N_position)
            MN_vector = N_position - metal_position
            print(MN_vector)
            # Get CNC atom ids.
            CNC_atom_ids = [N_atom_id]
            for bond in mol.bonds:
                if metal_atom.id in [bond.atom1.id, bond.atom2.id]:
                    continue
                if bond.atom1.id == N_atom_id:
                    CNC_atom_ids.append(bond.atom2.id)
                elif bond.atom2.id == N_atom_id:
                    CNC_atom_ids.append(bond.atom1.id)

            print(CNC_atom_ids)
            mol.write('temp.mol')
            # Get CNC plane.
            centroid = mol.get_centroid(atom_ids=CNC_atom_ids)
            CNC_plane_normal = mol.get_plane_normal(
                atom_ids=CNC_atom_ids
            )
            print(CNC_plane_normal)
            # Calculate angle between CNC plane and MN vector.
            pa = np.degrees(angle_between(MN_vector, CNC_plane_normal))
            print('oa', pa)
            plane_angles.append(pa)

        print(plane_angles)
        # Define the plane angle of a metal centre as the sum of all
        # plane angles of 4 coordinated atoms.
        plane_angle_avg = np.average([i for i in plane_angles])
        plane_angle_std = np.std([i for i in plane_angles])
        print('avg', plane_angle_avg, 'std', plane_angle_std)
        print('----')
        results['plane_angle_avg'].append(plane_angle_avg)
        results['plane_angle_std'].append(plane_angle_std)

    return results


def split_molecule(mol, N, fg_end, core=False, fg='bromine'):
    """
    Split a molecule into N molecules and add functional group.

    Parameters
    ----------
    mol : :class:`stk.Molecule`
        Molecule to split.

    N : :class:`int`
        Number of molecules to split into. Each will contain at least
        one :attr:`fg_end` and :attr:`fg`.

    fg_end : :class:`str`
        Functional group to search for as starting point.

    fg : :class:`str`, optional
        Functional group to append at split point. Defaults to
        'bromine'.

    Returns
    -------
    molecules : :class:`list` of :class:`stk.Molecule`
        N molecules.

    """
    molecules = []

    # Get number of fg_end.
    no_fg_end = 0
    if no_fg_end != N:
        raise ValueError(f'{N} {fg_end} were not found in molecule.')

    # For each fg_end, set a start atom.

    # Iterate through graph from

    if len(molecules) != N:
        raise ValueError(f'{N} molecules were not found.')

    return molecules


def calculate_NN_distance(bb, constructed=False, target_BB=None):
    """
    Calculate the N-N distance of ditopic building block.

    This function will not work for cages built from FGs other than
    metals + pyridine_N_metal.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    constructed : :class:`bool`
        `True` if bb is part of a ConstructedMolecule.
        Is so, Calculates the NN distance for all relavent building
        blocks.

    target_BB

    Returns
    -------
    NN_distance : :class:`float` or :class:`list` of :class:`float`
        Distance(s) between [angstrom] deleters in functional groups.

    """

    if constructed:
        if target_BB is None:
            target_bb_ids = [
                i for block in bb.building_block_counter
                for i in range(bb.building_block_counter[block])
            ]
        else:
            # Iterate over building blocks.
            target_bb_ids = []
            target_ident = target_BB.get_identity_key()
            id = 0
            for block in bb.building_block_counter:
                for i in range(bb.building_block_counter[block]):
                    ident = block.get_identity_key()
                    if ident == target_ident:
                        target_bb_ids.append(id)
                    id += 1

        # Only get properties if BB matches target_BB
        # (collect all if target_BB is None).

        # C building block id : [N atom ids]
        NN_pairings = {}
        for bond in bb.construction_bonds:
            if bond.atom1.atomic_number == 7:
                n_atom = bond.atom1
                other_atom = bond.atom2
            elif bond.atom2.atomic_number == 7:
                n_atom = bond.atom2
                other_atom = bond.atom1
            else:
                continue
            if other_atom.building_block_id not in target_bb_ids:
                continue
            if other_atom.building_block_id not in NN_pairings:
                NN_pairings[other_atom.building_block_id] = [n_atom.id]
            elif n_atom.id not in NN_pairings[
                other_atom.building_block_id
            ]:
                NN_pairings[other_atom.building_block_id].append(
                    n_atom.id
                )

        # Get N atoms that are part of functional groups that reacted
        # with the same BB.
        NN_distance = []
        for BB in NN_pairings:
            atom1_id, atom2_id = NN_pairings[BB]
            dist = bb.get_atom_distance(atom1_id, atom2_id)
            NN_distance.append(dist)
        return NN_distance
    else:
        fg_counts = 0
        N_positions = []
        for fg in bb.func_groups:
            if 'pyridine_N_metal' == fg.fg_type.name:
                fg_counts += 1
                # Get geometrical properties of the FG.
                # Get N position - deleter.
                N_position = bb.get_center_of_mass(
                    atom_ids=fg.get_deleter_ids()
                )
                N_positions.append(N_position)

        if fg_counts != 2:
            raise ValueError(
                f'{bb} does not have 2 pyridine_N_metal functional '
                'groups.'
            )

        # Calculate the angle between the two vectors.
        NN_distance = np.linalg.norm(N_positions[0] - N_positions[1])
        return NN_distance


def calculate_bite_angle(bb, constructed=False, target_BB=None):
    """
    Calculate the bite angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + pyridine_N_metal.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    constructed : :class:`bool`
        `True` if bb is part of a ConstructedMolecule.
        Is so, Calculates the bite angle for all relavent building
        blocks.

    target_BB

    Returns
    -------
    bite_angle : :class:`float` or :class:`list` of :class:`float`
        Angle(s) between two bonding vectors of molecule.

    """

    if constructed:
        if target_BB is None:
            target_bb_ids = [
                i for block in bb.building_block_counter
                for i in range(bb.building_block_counter[block])
            ]
        else:
            # Iterate over building blocks.
            target_bb_ids = []
            target_ident = target_BB.get_identity_key()
            id = 0
            for block in bb.building_block_counter:
                for i in range(bb.building_block_counter[block]):
                    ident = block.get_identity_key()
                    if ident == target_ident:
                        target_bb_ids.append(id)
                    id += 1

        # Only get properties if BB matches target_BB
        # (collect all if target_BB is None).

        # C building block id : [N atom ids]
        NN_pairings = {}
        # N_atom_ids : [bonded C ids]
        N_bonded_Cs = {}
        for bond in bb.construction_bonds:
            if bond.atom1.atomic_number == 7:
                n_atom = bond.atom1
                other_atom = bond.atom2
            elif bond.atom2.atomic_number == 7:
                n_atom = bond.atom2
                other_atom = bond.atom1
            else:
                continue

            if other_atom.building_block_id not in target_bb_ids:
                continue

            if other_atom.building_block_id not in NN_pairings:
                NN_pairings[other_atom.building_block_id] = [n_atom.id]
            elif n_atom.id not in NN_pairings[
                other_atom.building_block_id
            ]:
                NN_pairings[other_atom.building_block_id].append(
                    n_atom.id
                )

            if n_atom.id not in N_bonded_Cs:
                N_bonded_Cs[n_atom.id] = [other_atom.id]
            else:
                N_bonded_Cs[n_atom.id].append(other_atom.id)

        # Get N atoms that are part of functional groups that reacted
        # with the same BB.
        bite_angle = []
        for BB in NN_pairings:
            fg_counts = 0
            fg_vectors = []
            for n_atom_id in NN_pairings[BB]:
                fg_counts += 1
                # Get geometrical properties of the FG.
                # Get N position.
                N_position = bb.get_center_of_mass(
                    atom_ids=[n_atom_id]
                )

                # Get the ids of the C atoms bonded to N.
                bonded_C_ids = N_bonded_Cs[n_atom_id]

                # Get COM of neighbouring C atom positions - bonders.
                CC_MP = bb.get_center_of_mass(
                    atom_ids=bonded_C_ids
                )

                # Get vector between COM and N position.
                v = N_position - CC_MP
                fg_vectors.append(v)

            if fg_counts < 2:
                raise ValueError(
                    f'{bb} does not have 2 pyridine_N_metal '
                    'functional groups.'
                )
            bite_angle.append(abs(np.degrees(
                angle_between(*fg_vectors)
            )))

        return bite_angle
    else:
        fg_counts = 0
        fg_vectors = []
        for fg in bb.func_groups:
            if 'pyridine_N_metal' == fg.fg_type.name:
                fg_counts += 1
                # Get geometrical properties of the FG.
                # Get N position - deleter.
                N_position = bb.get_center_of_mass(
                    atom_ids=fg.get_deleter_ids()
                )
                # Get COM of neighbouring C atom positions - bonders.
                CC_MP = bb.get_center_of_mass(
                    atom_ids=fg.get_bonder_ids()
                )
                # Get vector between COM and N position.
                v = N_position - CC_MP
                fg_vectors.append(v)

        if fg_counts != 2:
            raise ValueError(
                f'{bb} does not have 2 pyridine_N_metal functional '
                'groups.'
            )

        # Calculate the angle between the two vectors.
        bite_angle = abs(np.degrees(angle_between(*fg_vectors)))
        return bite_angle


def filter_pyridine_FGs(mol):

    # Get all FG distances.
    fg_dists = sorted(
        [i for i in mol.get_bonder_distances()],
        key=lambda x: x[2],
        reverse=True
    )
    fgs_to_keep = [fg_dists[0][0], fg_dists[0][1]]
    mol.func_groups = tuple([
        mol.func_groups[i] for i, j in enumerate(mol.func_groups)
        if i in fgs_to_keep
    ])

    return mol


def calculate_ligand_distortion(
    mol,
    cage_name,
    free_ligand_name,
    free_NN_dists=None,
    free_bite_dists=None,
):
    """
    Calculate ligand distorion of ligands in mol.

    """

    if exists(free_ligand_name):
        free_ligand = stk.BuildingBlock.init_from_file(
            free_ligand_name,
            functional_groups=['pyridine_N_metal']
        )
        free_ligand = filter_pyridine_FGs(free_ligand)
    else:
        raise ValueError('need to build and opt all ligands!')

    # First measure is N-N distance.
    # Get N-N distance of all ligands in cage.
    cage_NNs = calculate_NN_distance(
        bb=mol,
        constructed=True,
        target_BB=free_ligand
    )

    if free_NN_dists is None:
        # Get N-N distance of free ligand.
        free_NN = calculate_NN_distance(
            bb=free_ligand,
            constructed=False
        )
    else:
        free_NN = np.mean(free_NN_dists)
    # Get average difference between NN in cage and free.
    NN_avg_cage_min_free = np.average([
        abs(i-free_NN) for i in cage_NNs
    ])

    # Second measure is bite angle.
    # Get bite angle of all ligands in cage.
    cage_bites = calculate_bite_angle(
        bb=mol,
        constructed=True,
        target_BB=free_ligand
    )

    if free_bite_dists is None:
        # Get bite angle of free ligand.
        free_bite = calculate_bite_angle(
            bb=free_ligand,
            constructed=False
        )
    else:
        free_bite = np.mean(free_bite_dists)
    # Get average difference between bite angle of each ligand and
    # free.
    bite_avg_cage_min_free = np.average([
        abs(i-free_bite) for i in cage_bites
    ])

    return NN_avg_cage_min_free, bite_avg_cage_min_free


def calculate_N_COM_N_angle(bb):
    """
    Calculate the N-COM-N angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + pyridine_N_metal.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    Returns
    -------
    angle : :class:`float`
        Angle between two bonding vectors of molecule.

    """

    fg_counts = 0
    fg_positions = []
    for fg in bb.func_groups:
        if 'pyridine_N_metal' == fg.fg_type.name:
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            N_position = bb.get_center_of_mass(
                atom_ids=fg.get_deleter_ids()
            )
            fg_positions.append(N_position)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have 2 pyridine_N_metal functional '
            'groups.'
        )

    # Get building block COM.
    COM_position = bb.get_center_of_mass()

    # Get vectors.
    fg_vectors = [i-COM_position for i in fg_positions]

    # Calculate the angle between the two vectors.
    angle = np.degrees(angle_between(*fg_vectors))
    return angle
