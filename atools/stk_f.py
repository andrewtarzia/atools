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
import logging
from itertools import combinations
import sys
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
        logging.info(f'{property} not defined')
        logging.info(f'possible properties: {properties}')
        sys.exit('exitting.')

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
        logging.info(f'properties not defined for {topology}')
        sys.exit('exitting')
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


def default_stk_MD_settings():
    """Default settings from stk source code as of 26/04/19.

    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 300,  # K
                'conformers': 50,
                'time_step': 1.0,  # fs
                'eq_time': 10,  # ps
                'simulation_time': 200,  # ps
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False
                }
    return Settings


def atarzia_short_MD_settings():
    """My default settings for short, crude cage optimizations in stk.


    Modified on 26/04/19.
    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 700,  # K
                'conformers': 50,
                'time_step': 1,  # fs
                'eq_time': 50,  # ps
                'simulation_time': 1000,  # ps -- 1 ns
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False}
    return Settings


def atarzia_long_MD_settings():
    """
    My default settings for rigorous cage optimizations in stk.

    Mimics: Computationally-inspired discovery of an unsymmetrical
    porous organic cage - DOI:10.1039/C8NR06868B

    Modified on 26/04/19.
    Modified on 06/06/19.
    """
    Settings = {'output_dir': None,
                'timeout': None,
                'force_field': 16,
                'temperature': 700,  # K
                'conformers': 5000,  # change from 10000
                'time_step': 0.5,  # fs
                'eq_time': 100,  # ps
                # ps -- 50 ns changed from 100 ns
                'simulation_time': -500,
                'maximum_iterations': 2500,
                'minimum_gradient': 0.05,
                'use_cache': False}
    return Settings


def optimize_structunit(infile, outfile, exec,
                        settings=None, method='OPLS'):
    """
    Read file into StructUnit and run optimization via method.

    """
    logging.info(f'loading in: {infile}')
    struct = stk.BuildingBlock.init_from_file(infile)
    if method == 'OPLS':
        # Use standard settings applied in andrew_marsh work if
        # md/settings is None.
        if settings is None:
            Settings = default_stk_MD_settings()
        else:
            Settings = settings
        logging.info(f'doing MD optimization of {infile}')
        # restricted=False optimization with OPLS forcefield by default
        ff = stk.MacroModelForceField(
            macromodel_path=exec, restricted=False
        )
        # MD process - run MD, collect N conformers, optimize each,
        # return lowest energy conformer
        md = stk.MacroModelMD(
            macromodel_path=exec,
            output_dir=Settings['output_dir'],
            timeout=Settings['timeout'],
            force_field=Settings['force_field'],
            temperature=Settings['temperature'],
            conformers=Settings['conformers'],
            time_step=Settings['time_step'],
            eq_time=Settings['eq_time'],
            simulation_time=Settings['simulation_time'],
            maximum_iterations=Settings['maximum_iterations'],
            minimum_gradient=Settings['minimum_gradient'],
            use_cache=Settings['use_cache']
        )
        macromodel = stk.OptimizerSequence(ff, md)
        macromodel.optimize(mol=struct)
        struct.write(outfile)
        logging.info('done')
    elif method == 'xtb':
        logging.info(f'doing xTB optimization of {infile}')
        xtb_opt = stk.XTB(
            xtb_path=exec,
            output_dir='xtb_opt',
            opt_level='tight',
            max_runs=1,
            calculate_hessian=False,
            unlimited_memory=True
        )
        xtb_opt.optimize(struct)
        struct.write(outfile)
        logging.info('done')
    else:
        logging.info(f'{method} is not implemented yet.')
        sys.exit('exitting')


def build_and_opt_cage(prefix, BB1, BB2, topology, macromod_,
                       settings=None, pdb=None, output_dir=None):
    """

    Keyword Arguments:
        prefix (str) - output file name prefix
        BB1 (str) - name of building block 1 file
        BB2 (str) - name of building block 2 file
        topology (stk.topology) = cage toplogy object
        macromod_ (str) - location of macromodel
        settings (dict) - settings for MacroModel Opt
        pdb (bool) - output PDB file of optimized cage
            (default is None)
        output_dir (str) - directory to save MacroModel output to
            (default is None)
    """
    # use standard settings applied in andrew_marsh work if md/settings
    # is None
    if settings is None:
        Settings = default_stk_MD_settings()
    else:
        Settings = settings

    # use default output dir (which is CWD) is output_dir is not given
    if output_dir is None:
        output_dir = Settings['output_dir']

    # try:
    cage = stk.ConstructedMolecule([BB1, BB2], topology)
    cage.write(prefix + '.mol')
    cage.dump(prefix + '.json')
    # restricted=True optimization with OPLS forcefield by default
    ff = stk.MacroModelForceField(
        macromodel_path=macromod_,
        restricted=True,
        output_dir=output_dir
    )
    # MD process - run MD, collect N conformers, optimize each,
    # return lowest energy conformer
    # no restricted
    md = stk.MacroModelMD(
        macromodel_path=macromod_,
        output_dir=output_dir,
        timeout=Settings['timeout'],
        force_field=Settings['force_field'],
        temperature=Settings['temperature'],
        conformers=Settings['conformers'],
        time_step=Settings['time_step'],
        eq_time=Settings['eq_time'],
        simulation_time=Settings['simulation_time'],
        maximum_iterations=Settings['maximum_iterations'],
        minimum_gradient=Settings['minimum_gradient'],
        use_cache=Settings['use_cache']
    )
    macromodel = stk.OptimizerSequence(ff, md)
    macromodel.optimize(mol=cage)
    cage.write(prefix + '_opt.mol')
    cage.dump(prefix + '_opt.json')
    if pdb is True:
        cage.write(prefix + '_opt.pdb')
    return cage


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
        'bond_lengths': [], 'angles': [],
        'torsions': [], 'plane_dev': []
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
        sys.exit(f'{N} {fg_end} were not found in molecule.')

    # For each fg_end, set a start atom.

    # Iterate through graph from

    if len(molecules) != N:
        sys.exit(f'{N} molecules were not found.')

    return molecules


def calculate_NN_distance(bb, constructed=False):
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

    Returns
    -------
    NN_distance : :class:`float` or :class:`list` of :class:`float`
        Distance(s) between [angstrom] deleters in functional groups.

    """

    if constructed:
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
            sys.exit(
                f'{bb} does not have 2 pyridine_N_metal functional '
                'groups.'
            )

        # Calculate the angle between the two vectors.
        NN_distance = np.linalg.norm(N_positions[0] - N_positions[1])
        return NN_distance


def calculate_bite_angle(bb, constructed=False):
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

    Returns
    -------
    bite_angle : :class:`float` or :class:`list` of :class:`float`
        Angle(s) between two bonding vectors of molecule.

    """

    if constructed:
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

            if fg_counts != 2:
                sys.exit(
                    f'{bb} does not have 2 pyridine_N_metal '
                    'functional groups.'
                )
            bite_angle.append(np.degrees(
                angle_between(*fg_vectors)
            ))

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
            sys.exit(
                f'{bb} does not have 2 pyridine_N_metal functional '
                'groups.'
            )

        # Calculate the angle between the two vectors.
        bite_angle = np.degrees(angle_between(*fg_vectors))
        return bite_angle


def calculate_ligand_distortion(mol, cage_name, free_ligand_name=None):
    """
    Calculate ligand distorion of ligands in mol.

    """
    if free_ligand_name is None:
        free_ligand_name = cage_name.replace('het_', '').split('_m')[0]
    if exists(f'{free_ligand_name}_opt.mol'):
        free_ligand = stk.BuildingBlock.init_from_file(
            f'{free_ligand_name}_opt.mol',
            functional_groups=['pyridine_N_metal']
        )
    else:
        sys.exit('need to build and opt all ligands!')
    # First measure is N-N distance.
    # Get N-N distance of all ligands in cage.
    cage_NNs = calculate_NN_distance(
        bb=mol,
        constructed=True
    )

    # Get N-N distance of free ligand.
    free_NN = calculate_NN_distance(
        bb=free_ligand,
        constructed=False
    )
    # Get average difference between NN in cage and free.
    NN_avg_cage_min_free = np.average([
        abs(i-free_NN) for i in cage_NNs
    ])

    # Second measure is bite angle.
    # Get bite angle of all ligands in cage.
    cage_bites = calculate_bite_angle(
        bb=mol,
        constructed=True
    )

    # Get bite angle of free ligand.
    free_bite = calculate_bite_angle(
        bb=free_ligand,
        constructed=False
    )
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
        sys.exit(
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
