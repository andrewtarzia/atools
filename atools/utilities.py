#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Utilities.

Author: Andrew Tarzia

Date Created: 29 May 2020
"""

from rdkit.Chem import AllChem as rdkit


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


def build_conformers(mol, N, ETKDG_version=None):
    """
    Convert stk mol into RDKit mol with N conformers.

    ETKDG_version allows the user to pick their choice of ETKDG params.

    `None` provides the settings used in ligand_combiner and unsymm.

    Other options:
        `v3`:
            New version from DOI: 10.1021/acs.jcim.0c00025
            with improved handling of macrocycles.

    """
    molecule = mol.to_rdkit_mol()
    molecule.RemoveAllConformers()

    if ETKDG_version is None:
        cids = rdkit.EmbedMultipleConfs(
            mol=molecule,
            numConfs=N,
            randomSeed=1000,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
            numThreads=4,
        )

    elif ETKDG_version == 'v3':
        params = rdkit.ETKDGv3()
        params.numThreads = 4
        params.randomSeed = 1000
        cids = rdkit.EmbedMultipleConfs(
            mol=molecule,
            numConfs=N,
            params=params
        )

    print(f'there are {molecule.GetNumConformers()} conformers')
    return cids, molecule
