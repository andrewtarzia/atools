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


def build_conformers(mol, N):
    """
    Convert stk mol into RDKit mol with N conformers.

    """
    molecule = mol.to_rdkit_mol()
    molecule.RemoveAllConformers()

    cids = rdkit.EmbedMultipleConfs(
        mol=molecule,
        numConfs=N,
        randomSeed=1000,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True
    )
    print(f'there are {molecule.GetNumConformers()} conformers')
    return cids, molecule
