#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Functions that are useful for stk usage.

Author: Andrew Tarzia

Date Created: 18 Mar 2019
"""

from os.path import exists
from os import mkdir
import stk
import numpy as np
from mendeleev import element
import networkx as nx
from scipy.spatial.distance import euclidean

from .IO_tools import read_gfnx2xtb_eyfile
from .stko_f import optimize_conformer, calculate_energy
from .utilities import build_conformers, update_from_rdkit_conf


class AromaticCNCFactory(stk.FunctionalGroupFactory):
    """
    A subclass of stk.SmartsFunctionalGroupFactory.

    """

    def __init__(self, bonders=(1, ), deleters=()):
        """
        Initialise :class:`.AromaticCNCFactory`

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
            yield AromaticCNC(
                carbon1=atoms[0],
                nitrogen=atoms[1],
                carbon2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class AromaticCNC(stk.GenericFunctionalGroup):
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


class AromaticCNNFactory(stk.FunctionalGroupFactory):
    """
    A subclass of stk.SmartsFunctionalGroupFactory.

    """

    def __init__(self, bonders=(1, ), deleters=()):
        """
        Initialise :class:`.AromaticCNNFactory`

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
            yield AromaticCNN(
                carbon=atoms[0],
                nitrogen=atoms[1],
                nitrogen2=atoms[2],
                bonders=tuple(atoms[i] for i in self._bonders),
                deleters=tuple(atoms[i] for i in self._deleters),
            )


class AromaticCNN(stk.GenericFunctionalGroup):
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


def get_stk_bond_angle(mol, atom1_id, atom2_id, atom3_id):
    # TODO: Fill in the doc string for this including defintiions.
    atom1_pos, = mol.get_atomic_positions(atom_ids=atom1_id)
    atom2_pos, = mol.get_atomic_positions(atom_ids=atom2_id)
    atom3_pos, = mol.get_atomic_positions(atom_ids=atom3_id)
    v1 = atom1_pos - atom2_pos
    v2 = atom3_pos - atom2_pos
    return stk.vector_angle(v1, v2)


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


def get_atom_distance(molecule, atom1_id, atom2_id):
    """
    Return the distance between atom1 and atom2.

    Parameters
    ----------
    molecule : :class:`stk.Molecule`

    atom1_id : :class:`int`
        The id of atom1.

    atom2_id : :class:`int`
        The id of atom2.

    Returns
    -------
    :class:`float`
        The euclidean distance between two atoms.

    """

    position_matrix = molecule.get_position_matrix()

    distance = euclidean(
        u=position_matrix[atom1_id],
        v=position_matrix[atom2_id]
    )

    return float(distance)


def get_organic_linkers(cage, metal_atom_nos, file_prefix=None):
    """
    Extract a list of organic linker .Molecules from a cage.

    Parameters
    ----------
    cage : :class:`stk.Molecule`
        Molecule to get the organic linkers from.

    metal_atom_nos : :class:`iterable` of :class:`int`
        The atomic number of metal atoms to remove from structure.

    file_prefix : :class:`str`, optional
        Prefix to file name of each output ligand structure.
        Eventual file name is:
        "file_prefix"{number of atoms}_{idx}_{i}.mol
        Where `idx` determines if a molecule is unique by smiles.

    Returns
    -------
    org_lig : :class:`dict` of :class:`stk.BuildingBlock`
        Dictionary of building blocks where the key is the file name,
        and the value is the stk building block.

    smiles_keys : :class:`dict` of :class:`int`
        Key is the linker smiles, value is the idx of that smiles.

    """

    org_lig = {}

    # Produce a graph from the cage that does not include metals.
    cage_g = nx.Graph()
    atom_ids_in_G = set()
    for atom in cage.get_atoms():
        if atom.get_atomic_number() in metal_atom_nos:
            continue
        cage_g.add_node(atom)
        atom_ids_in_G.add(atom.get_id())

    # Add edges.
    for bond in cage.get_bonds():
        a1id = bond.get_atom1().get_id()
        a2id = bond.get_atom2().get_id()
        if a1id in atom_ids_in_G and a2id in atom_ids_in_G:
            cage_g.add_edge(bond.get_atom1(), bond.get_atom2())

    # Get disconnected subgraphs as molecules.
    # Sort and sort atom ids to ensure molecules are read by RDKIT
    # correctly.
    connected_graphs = [
        sorted(subgraph, key=lambda a: a.get_id())
        for subgraph in sorted(nx.connected_components(cage_g))
    ]
    smiles_keys = {}
    for i, cg in enumerate(connected_graphs):
        # Get atoms from nodes.
        atoms = list(cg)
        atom_ids = [i.get_id() for i in atoms]
        cage.write(
            'temporary_linker.mol',
            atom_ids=atom_ids
        )
        temporary_linker = stk.BuildingBlock.init_from_file(
            'temporary_linker.mol'
        ).with_canonical_atom_ordering()
        smiles_key = stk.Smiles().get_key(temporary_linker)
        if smiles_key not in smiles_keys:
            smiles_keys[smiles_key] = len(smiles_keys.values())+1
        idx = smiles_keys[smiles_key]
        sgt = str(len(atoms))
        # Write to mol file.
        if file_prefix is None:
            filename_ = f'organic_linker_s{sgt}_{idx}_{i}.mol'
        else:
            filename_ = f'{file_prefix}{sgt}_{idx}_{i}.mol'

        org_lig[filename_] = temporary_linker
        # Rewrite to fix atom ids.
        org_lig[filename_].write(filename_)
        org_lig[filename_] = stk.BuildingBlock.init_from_file(
            filename_
        )

    return org_lig, smiles_keys


def get_lowest_energy_conformers(
    org_ligs,
    smiles_keys,
    file_prefix=None
):
    """
    Determine the lowest energy conformer of cage organic linkers.

    Will do multiple if there are multiple types.

    Parameters
    ----------
    org_ligs : :class:`dict` of :class:`stk.BuildingBlock`
        Dictionary of building blocks where the key is the file name,
        and the value is the stk building block.

    smiles_keys : :class:`dict` of :class:`int`
        Key is the linker smiles, value is the idx of that smiles.

    file_prefix : :class:`str`, optional
        Prefix to file name of each output ligand structure.
        Eventual file name is:
        "file_prefix"{number of atoms}_{idx}_{i}.mol
        Where `idx` determines if a molecule is unique by smiles.

    """

    for lig in org_ligs:
        stk_lig = org_ligs[lig]
        smiles_key = stk.Smiles().get_key(stk_lig)
        idx = smiles_keys[smiles_key]
        sgt = str(stk_lig.get_num_atoms())
        # Get optimized ligand name that excludes any cage information.
        if file_prefix is None:
            filename_ = f'organic_linker_s{sgt}_{idx}_opt.mol'
            ligand_name_ = f'organic_linker_s{sgt}_{idx}_opt'
        else:
            filename_ = f'{file_prefix}{sgt}_{idx}_opt.mol'
            ligand_name_ = f'{file_prefix}{sgt}_{idx}_opt'

        if not exists(filename_):
            if not exists(f'{ligand_name_}_confs/'):
                mkdir(f'{ligand_name_}_confs/')
            low_e_conf = get_lowest_energy_conformer(
                name=ligand_name_,
                mol=stk_lig
            )
            low_e_conf.write(filename_)


def get_lowest_energy_conformer(
    name,
    mol,
    opt_level='extreme',
    charge=0,
    no_unpaired_e=0,
    max_runs=1,
    calc_hessian=False,
    solvent=None
):
    """
    Get lowest energy conformer of molecule.

    Method:
        1) ETKDG conformer search on molecule
        2) xTB normal optimisation of each conformer
        3) xTB opt_level optimisation of lowest energy conformer
        4) save file

    """

    # Run ETKDG on molecule.
    print(f'....running ETKDG on {name}')
    cids, confs = build_conformers(mol, N=100)

    # Optimize all conformers at normal level with xTB.
    low_e_conf_id = -100
    low_e = 10E20
    for cid in cids:
        name_ = f'{name}_confs/c_{cid}'
        ey_file = f'{name}_confs/c_{cid}_eyout'
        print(name, ey_file)
        mol = update_from_rdkit_conf(
            mol,
            confs,
            conf_id=cid
        )
        mol.write(f'temp_c_{cid}.mol')

        # Optimize.
        opt_mol = optimize_conformer(
            name=name_+'_opt',
            mol=mol,
            opt_level='normal',
            charge=charge,
            no_unpaired_e=no_unpaired_e,
            max_runs=max_runs,
            calc_hessian=calc_hessian,
            solvent=solvent
        )

        # Get energy.
        calculate_energy(
            name=name_+'_ey',
            mol=opt_mol,
            ey_file=ey_file,
            charge=charge,
            no_unpaired_e=no_unpaired_e,
            solvent=solvent
        )
        ey = read_gfnx2xtb_eyfile(ey_file)
        if ey < low_e:
            low_e_conf_id = cid
            low_e = ey
        print(ey, low_e, low_e_conf_id, cid)

    # Get lowest energy conformer.
    low_e_conf = update_from_rdkit_conf(
        mol,
        confs,
        conf_id=low_e_conf_id
    )
    low_e_conf.write('temp_pre_opt.mol')

    # Optimize lowest energy conformer at opt_level.
    low_e_conf = optimize_conformer(
        name=name_+'low_e_opt',
        mol=low_e_conf,
        opt_level=opt_level,
        charge=charge,
        no_unpaired_e=no_unpaired_e,
        max_runs=max_runs,
        calc_hessian=calc_hessian,
        solvent=solvent
    )
    low_e_conf.write('temp_post_opt.mol')
    print(low_e_conf_id)

    # Return molecule.
    return low_e_conf
