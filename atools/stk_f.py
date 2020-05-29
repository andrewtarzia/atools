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
from mendeleev import element
import networkx as nx
from scipy.spatial.distance import euclidean

from .calculations import (
    get_dihedral,
    shortest_distance_to_plane,
    angle_between
)
from .pymatgen_f import calculate_sites_order_values
from .IO_tools import convert_stk_to_pymatgen


class NPyridineFactory(stk.FunctionalGroupFactory):
    """
    A subclass of stk.SmartsFunctionalGroupFactory.

    """

    def __init__(self, bonders=(1, ), deleters=()):
        """
        Initialise :class:`.NPyridineFactory`

        """

        self._bonders = bonders
        self._deleters = deleters

    def get_functional_groups(self, molecule):
        generic_functional_groups = stk.SmartsFunctionalGroupFactory(
            smarts='[#6]~[#7X2]~[#6]',
            bonders=self._bonders,
            deleters=self._deleters
        ).get_functional_groups(molecule)
        for fg in generic_functional_groups:
            atom_ids = (i.get_id() for i in fg.get_atoms())
            atoms = tuple(molecule.get_atoms(atom_ids))
            yield NPyridine(
                carbon1=atoms[0],
                nitrogen=atoms[1],
                carbon2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class NPyridine(stk.GenericFunctionalGroup):
    """
    Represents an N atom in pyridine functional group.

    The structure of the functional group is given by the pseudo-SMILES
    ``[carbon][nitrogen][carbon]``.

    """

    def __init__(self, carbon1, nitrogen, carbon2, bonders, deleters):
        """
        Initialize a :class:`.Alcohol` instance.

        Parameters
        ----------
        carbon1 : :class:`.C`
            The first carbon atom.

        nitrogen : :class:`.N`
            The nitrogen atom.

        carbon2 : :class:`.C`
            The second carbon atom.

        bonders : :class:`tuple` of :class:`.Atom`
            The bonder atoms.

        deleters : :class:`tuple` of :class:`.Atom`
            The deleter atoms.

        """

        self._carbon1 = carbon1
        self._nitrogen = nitrogen
        self._carbon2 = carbon2
        atoms = (carbon1, nitrogen, carbon2)
        super().__init__(atoms, bonders, deleters)

    def get_carbon1(self):
        """
        Get the first carbon atom.

        Returns
        -------
        :class:`.C`
            The first carbon atom.

        """

        return self._carbon1

    def get_carbon2(self):
        """
        Get the second carbon atom.

        Returns
        -------
        :class:`.C`
            The second carbon atom.

        """

        return self._carbon2

    def get_nitrogen(self):
        """
        Get the nitrogen atom.

        Returns
        -------
        :class:`.N`
            The nitrogen atom.

        """

        return self._nitrogen

    def clone(self):
        clone = super().clone()
        clone._carbon1 = self._carbon1
        clone._nitrogen = self._nitrogen
        clone._carbon2 = self._carbon2
        return clone

    def with_atoms(self, atom_map):
        clone = super().with_atoms(atom_map)
        clone._carbon1 = atom_map.get(
            self._carbon1.get_id(),
            self._carbon1,
        )
        clone._nitrogen = atom_map.get(
            self._nitrogen.get_id(),
            self._nitrogen,
        )
        clone._carbon2 = atom_map.get(
            self._carbon2.get_id(),
            self._carbon2,
        )
        return clone

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self._carbon1}, {self._nitrogen}, {self._carbon2}, '
            f'bonders={self._bonders})'
        )


class NTriazoleFactory(stk.FunctionalGroupFactory):
    """
    A subclass of stk.SmartsFunctionalGroupFactory.

    """

    def __init__(self, bonders=(1, ), deleters=()):
        """
        Initialise :class:`.NTriazoleFactory`

        """

        self._bonders = bonders
        self._deleters = deleters

    def get_functional_groups(self, molecule):
        generic_functional_groups = stk.SmartsFunctionalGroupFactory(
            smarts='[#6]~[#7X2]~[#7X2]',
            bonders=self._bonders,
            deleters=self._deleters
        ).get_functional_groups(molecule)
        for fg in generic_functional_groups:
            atom_ids = (i.get_id() for i in fg.get_atoms())
            atoms = tuple(molecule.get_atoms(atom_ids))
            yield NTriazole(
                carbon=atoms[0],
                nitrogen=atoms[1],
                nitrogen2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class NTriazole(stk.GenericFunctionalGroup):
    """
    Represents an N atom in pyridine functional group.

    The structure of the functional group is given by the pseudo-SMILES
    ``[carbon][nitrogen][nitrogen]``.

    """

    def __init__(
        self,
        carbon,
        nitrogen,
        nitrogen2,
        bonders,
        deleters
    ):
        """
        Initialize a :class:`.Alcohol` instance.

        Parameters
        ----------
        carbon : :class:`.C`
            The carbon atom.

        nitrogen : :class:`.N`
            The first and bonding (default) nitrogen atom.

        nitrogen2 : :class:`.C`
            The second nitrogen atom.

        bonders : :class:`tuple` of :class:`.Atom`
            The bonder atoms.

        deleters : :class:`tuple` of :class:`.Atom`
            The deleter atoms.

        """

        self._carbon = carbon
        self._nitrogen = nitrogen
        self._nitrogen2 = nitrogen2
        atoms = (carbon, nitrogen, nitrogen2)
        super().__init__(atoms, bonders, deleters)

    def get_carbon(self):
        """
        Get the carbon atom.

        Returns
        -------
        :class:`.C`
            The carbon atom.

        """

        return self._carbon

    def get_nitrogen2(self):
        """
        Get the second nitrogen atom.

        Returns
        -------
        :class:`.N`
            The second nitrogen atom.

        """

        return self._nitrogen2

    def get_nitrogen(self):
        """
        Get the first nitrogen atom.

        Returns
        -------
        :class:`.N`
            The first nitrogen atom.

        """

        return self._nitrogen

    def clone(self):
        clone = super().clone()
        clone._carbon = self._carbon
        clone._nitrogen = self._nitrogen
        clone._nitrogen2 = self._nitrogen2
        return clone

    def with_atoms(self, atom_map):
        clone = super().with_atoms(atom_map)
        clone._carbon = atom_map.get(
            self._carbon.get_id(),
            self._carbon,
        )
        clone._nitrogen = atom_map.get(
            self._nitrogen.get_id(),
            self._nitrogen,
        )
        clone._nitrogen2 = atom_map.get(
            self._nitrogen2.get_id(),
            self._nitrogen2,
        )
        return clone

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self._carbon}, {self._nitrogen}, {self._nitrogen2}, '
            f'bonders={self._bonders})'
        )


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

    # TODO: Fill in the doc string for this including defintiions.
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

    # TODO: Fill in the doc string for this including defintiions.
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


def update_from_rdkit_conf(stk_mol, rdk_mol, conf_id):
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

    pos_mat = rdk_mol.GetConformer(id=conf_id).GetPositions()
    return stk_mol.with_position_matrix(pos_mat)


def get_stk_bond_angle(mol, atom1_id, atom2_id, atom3_id):
    # TODO: Fill in the doc string for this including defintiions.
    atom1_pos, = mol.get_atomic_positions(atom_ids=atom1_id)
    atom2_pos, = mol.get_atomic_positions(atom_ids=atom2_id)
    atom3_pos, = mol.get_atomic_positions(atom_ids=atom3_id)
    v1 = atom1_pos - atom2_pos
    v2 = atom3_pos - atom2_pos
    return stk.vector_angle(v1, v2)


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
        Defulats to False. True if the OPs for each site are desired.

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
    for atom in mol.atoms:
        if atom.atomic_number == metal:
            sites.append(atom.id)
            bonds = [
                i for i in mol.bonds
                if i.atom1.id == atom.id or i.atom2.id == atom.id
            ]
            a_neigh = []
            for b in bonds:
                if b.atom1.id == atom.id:
                    a_neigh.append(b.atom2.id)
                elif b.atom2.id == atom.id:
                    a_neigh.append(b.atom1.id)
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
            mol.get_atom_distance(
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
            i for i in mol.get_atom_positions(atom_ids=torsion_ids)
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

    # TODO: Finish this function.
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


def get_target_bb_ids(bb, target_BB=None):
    """
    Determine the ID of buidling blocks matching target_BB.

    """
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

    return target_bb_ids


def calculate_ligand_energy(
    bb,
    constructed=False,
    target_BB=None
):
def get_center_of_mass(molecule, atom_ids=None):
    """
    Calculate the ligand energy of bb.
    Return the centre of mass.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    constructed : :class:`bool`
        `True` if bb is part of a ConstructedMolecule.
        Is so, Calculates the strain energy for all relavent building
        blocks.
    molecule : :class:`stk.Molecule`

    target_BB
    atom_ids : :class:`iterable` of :class:`int`, optional
        The ids of atoms which should be used to calculate the
        center of mass. If ``None``, then all atoms will be used.

    Returns
    -------
    energies : :class:`list` of :class:`float`
        Energies of all bb in ConstructedMolecule if
        :attr:`contructed` is `True` or energy of bb.
        Units: kJ/mol.

    """
    :class:`numpy.ndarray`
        The coordinates of the center of mass.

    if constructed:
        target_bb_ids = get_target_bb_ids(bb, target_BB)
    References
    ----------
    https://en.wikipedia.org/wiki/Center_of_mass

        # Only get properties if BB matches target_BB
        # (collect all if target_BB is None).
    """

        # Get BB coordinates and bonds for each target_bb_id.
        BB_properties = []

        # Iterate through BB properties and output a structure to
        # calculate the energy of.
        BB_energies = []
        for BB in BB_properties:
            BB_mol = convert_BB_to_mol(BB)
            BB_energy = get_BB_energy(BB_mol)
            BB_energies.append(BB_energy)
        return BB_energies
    else:
        BB_energy = get_BB_energy(bb)
        BB_energies = [BB_energy]
        return BB_energies
    if atom_ids is None:
        atom_ids = range(molecule.get_num_atoms())
    elif not isinstance(atom_ids, (list, tuple)):
        # Iterable gets used twice, once in get_atom_positions
        # and once in zip.
        atom_ids = list(atom_ids)

    center = 0
    total_mass = 0.
    coords = molecule.get_atomic_positions(atom_ids)
    atoms = molecule.get_atoms(atom_ids)
    for atom, coord in zip(atoms, coords):
        mass = element(atom.__class__.__name__).atomic_weight
        total_mass += mass
        center += mass*coord
    return np.divide(center, total_mass)

def calculate_NN_distance(bb, constructed=False, target_BB=None):
    """
    Calculate the N-N distance of ditopic building block.

    This function will not work for cages built from FGs other than
    metals + NPyridine and metals + NTriazole.
def get_atom_distance(molecule, atom1_id, atom2_id):
    """
    Return the distance between atom1 and atom2.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.
    molecule : :class:`stk.Molecule`

    constructed : :class:`bool`
        `True` if bb is part of a ConstructedMolecule.
        Is so, Calculates the NN distance for all relavent building
        blocks.
    atom1_id : :class:`int`
        The id of atom1.

    target_BB
    atom2_id : :class:`int`
        The id of atom2.

    Returns
    -------
    NN_distance : :class:`float` or :class:`list` of :class:`float`
        Distance(s) between [angstrom] deleters in functional groups.
    :class:`float`
        The euclidean distance between two atoms.

    """

    if constructed:
        target_bb_ids = get_target_bb_ids(bb, target_BB)

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
        for fg in bb.get_functional_groups():
            if isinstance(fg, NPyridine) or isinstance(fg, NTriazole):
                fg_counts += 1
                # Get geometrical properties of the FG.
                # Get N position - deleter.
                N_position, = bb.get_atomic_positions(
                    atom_ids=fg.get_nitrogen().get_id()
                )
                N_positions.append(N_position)

        if fg_counts != 2:
            raise ValueError(
                f'{bb} does not have 2 NPyridine or NTriazole '
                'functional groups.'
            )

        # Calculate the angle between the two vectors.
        NN_distance = np.linalg.norm(N_positions[0] - N_positions[1])
        return NN_distance
    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id]
    )

def calculate_bite_angle(bb, constructed=False, target_BB=None):
    """
    Calculate the bite angle of a ditopic building block.

    Here the bite angle is defined `visually` as in:
        https://doi.org/10.1016/j.ccr.2018.06.010
    return float(distance)

    It is calculated using the angles between binding vectors and the
    N to N vector.

    This function will not work for cages built from FGs other than
    metals + NPyridine and metals + NTriazole.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock` or :class:`stk.ConstructedMolecule`
        Stk molecule to analyse.

    constructed : :class:`bool`
        `True` if bb is part of a :class:`stk.ConstructedMolecule`.
        Is so, Calculates the bite angle for all relavent building
        blocks.

    target_BB : :class:`stk.BuildingBlock`
        Stk building block to calculate bite angle of in a
        :class:`stk.ConstructedMolecule`. Searches for BuidlingBlocks
        in :class:`stk.ConstructedMolecule` with the same Identity Key.

    Returns
    -------
    bite_angle : :class:`float` or :class:`list` of :class:`float`
        Angle(s) between two bonding vectors of molecule.

    """

    if constructed:
        target_bb_ids = get_target_bb_ids(bb, target_BB)

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
        N_positions = []
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
                N_positions.append(N_position)

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
                    f'{bb} does not have 2 NPyridine '
                    'functional groups.'
                )
            # Get N to N vector.
            NN_vec = N_positions[1] - N_positions[0]

            # Calculate the angle between the two vectors.
            angle_1 = np.degrees(angle_between(
                fg_vectors[0], NN_vec
            ))
            angle_2 = np.degrees(angle_between(
                fg_vectors[1], -NN_vec
            ))
            bite_angle.append((angle_1 - 90) + (angle_2 - 90))

        return bite_angle
    else:
        fg_counts = 0
        fg_vectors = []
        N_positions = []

        for fg in bb.get_functional_groups():
            if isinstance(fg, NPyridine) or isinstance(fg, NTriazole):
                fg_counts += 1
                # Get geometrical properties of the FG.
                # Get N position.
                N_position, = bb.get_atomic_positions(
                    atom_ids=fg.get_nitrogen().get_id()
                )
                N_positions.append(N_position)
                # Get centroid of neighbouring C atom positions.
                CC_MP = bb.get_centroid(
                    atom_ids=(
                        fg.get_carbon1().get_id(),
                        fg.get_carbon2().get_id()
                    )
                )
                # Get vector between COM and N position.
                v = N_position - CC_MP
                fg_vectors.append(v)

        if fg_counts != 2:
            raise ValueError(
                f'{bb} does not have exactly 2 NPyridine or NTriazole '
                'functional groups.'
            )

        # Get N to N vector.
        NN_vec = N_positions[1] - N_positions[0]

        # Calculate the angle between the two vectors.
        angle_1 = np.degrees(angle_between(
            fg_vectors[0], NN_vec
        ))
        angle_2 = np.degrees(angle_between(
            fg_vectors[1], -NN_vec
        ))
        bite_angle = (angle_1 - 90) + (angle_2 - 90)
        return bite_angle


def get_furthest_pair_FGs(stk_mol):
    """
    Returns the pair of functional groups that are furthest apart.

    """

    if stk_mol.get_num_functional_groups() == 2:
        return tuple(i for i in stk_mol.get_functional_groups())
    elif stk_mol.get_num_functional_groups() < 2:
        raise ValueError(f'{stk_mol} does not have at least 2 FGs')

    fg_centroids = [
        (fg, stk_mol.get_centroid(atom_ids=fg.get_placer_ids()))
        for fg in stk_mol.get_functional_groups()
    ]

    fg_dists = sorted(
        [
            (i[0], j[0], euclidean(i[1], j[1]))
            for i, j in combinations(fg_centroids, 2)
        ],
        key=lambda x: x[2],
        reverse=True
    )

    return (fg_dists[0][0], fg_dists[0][1])


def filter_pyridine_FGs(mol):
    # TODO: Fill in the doc string for this including defintiions.

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
    free_energy_dists=None
):
    """
    Calculate ligand distorion of ligands in mol.


    Strain energy definition:
    :attr:`contructed` is `True`:
        strain energy = E(bb extracted from cage) -
                        E(lowest energy conformer of bb)

    :attr:`contructed` is `False`
        strain energy = E(bb) - E(lowest energy conformer of bb)

    # TODO: Fill in the doc string for this including defintiions.

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

    # TODO: Finish this!
    # free_lig_energy = get_lowest_energy_conformer(bb=free_ligand)
    #
    # lig_energies = calculate_ligand_energy(
    #     bb=mol,
    #     constructed=True,
    #     target_BB=free_ligand
    # )
    # strain_energies = [i-free_lig_energy for i in lig_energies]
    # sum_strain_energy = sum(strain_energies)
    # print(strain_energies, sum_strain_energy)
    # import sys
    # sys.exit()

    distortions = (
        NN_avg_cage_min_free,
        bite_avg_cage_min_free,
        # sum_strain_energy
    )

    return distortions


def calculate_N_COM_N_angle(bb):
    """
    Calculate the N-COM-N angle of a ditopic building block.

    This function will not work for cages built from FGs other than
    metals + NPyridine and metals + NTriazole.

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
    for fg in bb.get_functional_groups():
        if isinstance(fg, NPyridine) or isinstance(fg, NTriazole):
            fg_counts += 1
            # Get geometrical properties of the FG.
            # Get N position - deleter.
            N_position, = bb.get_atomic_positions(
                atom_ids=fg.get_nitrogen().get_id()
            )
            fg_positions.append(N_position)

    if fg_counts != 2:
        raise ValueError(
            f'{bb} does not have 2 NPyridine or NTriazole functional '
            'groups.'
        )

    # Get building block COM.
    COM_position = get_center_of_mass(bb)

    # Get vectors.
    fg_vectors = [i-COM_position for i in fg_positions]

    # Calculate the angle between the two vectors.
    angle = np.degrees(angle_between(*fg_vectors))
    return angle


def get_BB_energy(bb, file_prefix, energy_type='total'):
    """
    Get the GFN2-xTB energy of bb.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    file_prefix : :class:`str`
        Prefix to file names that will be output in this process.
        Produces with this prefix:
            `_lowE_conf.json` : Json containing energy values.
            `_lowE_conf.mol` : MolFile containing structure.
            Associated xtb files.

    energy_type : :class:`str`
        Type of energy to return.

    Returns
    -------
    energy :class:`float`
        Desired energy value in kJ/mol.

    """


def get_lowest_energy_conformer(bb, file_prefix, energy_type='total'):
    """
    Get the lowest energy conformer of bb using GFN2-xTB and RDKit.

    Parameters
    ----------
    bb : :class:`stk.BuildingBlock`
        stk molecule to analyse.

    file_prefix : :class:`str`
        Prefix to file names that will be output in this process.
        Produces with this prefix:
            `_lowE_conf.json` : Json containing energy values.
            `_lowE_conf.mol` : MolFile containing structure.
            Associated xtb files.

    energy_type : :class:`str`
        Type of energy to return.

    Returns
    -------
    energy :class:`float`
        Desired energy value in kJ/mol.

    """

    if exists(f'{file_prefix}_lowE_conf.json'):
        # Extract energy from JSON.
        energy = 0
    else:
        # Run calculation.
        energy = 0
        BB_energy = get_BB_energy(bb)

    return energy


def get_center_of_mass(molecule, atom_ids=None):
    """
    Return the centre of mass.

    Parameters
    ----------
    molecule : :class:`stk.Molecule`

    atom_ids : :class:`iterable` of :class:`int`, optional
        The ids of atoms which should be used to calculate the
        center of mass. If ``None``, then all atoms will be used.

    Returns
    -------
    :class:`numpy.ndarray`
        The coordinates of the center of mass.

    References
    ----------
    https://en.wikipedia.org/wiki/Center_of_mass

    """

    if atom_ids is None:
        atom_ids = range(molecule.get_num_atoms())
    elif not isinstance(atom_ids, (list, tuple)):
        # Iterable gets used twice, once in get_atom_positions
        # and once in zip.
        atom_ids = list(atom_ids)

    center = 0
    total_mass = 0.
    coords = molecule.get_atomic_positions(atom_ids)
    atoms = molecule.get_atoms(atom_ids)
    for atom, coord in zip(atoms, coords):
        mass = element(atom.__class__.__name__).atomic_weight
        total_mass += mass
        center += mass*coord
    return np.divide(center, total_mass)
